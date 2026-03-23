#!/usr/bin/env python3
"""
作者：zhao
环境：dustmap
日期：2025.11.12
使用方法：
python add_3D_dust_map.py
requirements:
pip install git+https://github.com/SunnyHina/dustmaps3d.git
pip install tables
参考文章：Wang et al. (2025), An all-sky 3D dust map based on Gaia and LAMOST
网站：https://nadc.china-vo.org/data/dustmaps/calculator
输入文件：我们的原始数据文件，文件中只需要有final_pos就可以查询extinction
使用方法：python add_dust_map.py all_dwd_clean.hdf5
作用：在我们原来的数据结构中加入银道坐标系以及E（B—V）的数值以及视星等的6列数据
"""
import os
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord
from dustmaps3d import dustmaps3d
from dustmaps3d import dustmaps3d_from_df
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
import contextlib
import io
import shutil
import gc
warnings.filterwarnings('ignore')

class SingleBatchProgressProcessor:
    def __init__(self, n_cores=4):
        """
        初始化处理器
        
        参数:
        n_cores: 三个步骤共同使用的CPU核心数
        """
        self.n_cores = min(n_cores, cpu_count())
        self.extinction_coefficients = {
            'u': 4.81, 'g': 3.64, 'r': 2.70, 'i': 2.06, 'z': 1.58, 'y': 1.31
        }
        mp.set_start_method('spawn', force=True)
        
        print(f"三个处理步骤将共同使用 {self.n_cores} 个CPU核心")

    def _add_coordinates_to_df(self, df):
        """为DataFrame添加坐标列（向量化，一次性处理）"""
        try:
            # 检查必要列
            if 'x_pc' not in df.columns:
                print("  警告: 缺少 x_pc 列，跳过坐标转换")
                return df
            
            # 向量化坐标转换（直接除法，避免astropy单位转换开销）
            x_pc = df['x_pc'].values.astype(float)
            y_pc = df['y_pc'].values.astype(float)
            z_pc = df['z_pc'].values.astype(float)
            
            # 转换为kpc（直接除以1000，快很多）
            x_kpc = x_pc / 1000.0
            y_kpc = y_pc / 1000.0
            z_kpc = z_pc / 1000.0
            
            # 批量坐标转换
            galactocentric_coords = coord.Galactocentric(
                x=x_kpc * u.kpc, y=y_kpc * u.kpc, z=z_kpc * u.kpc,
                galcen_distance=8.5 * u.kpc, z_sun=0 * u.pc
            )
            
            try:
                galactic_coords = galactocentric_coords.transform_to(coord.Galactic())
            except Exception:
                icrs_coords = galactocentric_coords.transform_to(coord.ICRS())
                galactic_coords = icrs_coords.transform_to(coord.Galactic())
            
            # 添加新列
            df['l'] = galactic_coords.l.deg
            df['b'] = galactic_coords.b.deg
            df['distance_to_sun_pc'] = galactic_coords.distance.to(u.pc).value
            df['d'] = df['distance_to_sun_pc'] / 1000.0
            
            # 验证距离是否有效
            invalid = (df['distance_to_sun_pc'] <= 0) | np.isnan(df['distance_to_sun_pc'])
            if np.any(invalid):
                df.loc[invalid, 'distance_to_sun_pc'] = np.nan
                df.loc[invalid, 'd'] = np.nan
            
            return df
            
        except Exception as e:
            print(f"坐标转换错误: {e}")
            return df

    def _add_extinction_to_df(self, df):
        """为DataFrame添加消光值列（一次性查询所有点）"""
        try:
            # 检查是否已有坐标列
            if 'l' not in df.columns or 'b' not in df.columns or 'd' not in df.columns:
                df['EBV'] = 0.0
                return df
            
            # 检查是否有有效坐标
            valid_coords = (~df['l'].isna()) & (~df['b'].isna()) & (~df['d'].isna())
            df['EBV'] = 0.0
            
            if np.any(valid_coords):
                # 收集所有有效坐标
                dustmaps_input = df.loc[valid_coords, ['l', 'b', 'd']].copy()
                dustmaps_input = dustmaps_input.reset_index(drop=True)
                
                # 一次性查询所有消光值
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    processed_dustmaps = dustmaps3d_from_df(
                        dustmaps_input, 
                        n_process=self.n_cores,
                        chunk_size=max(1000, len(dustmaps_input) // self.n_cores)
                    )
                
                # 将消光值填入对应的行
                ebv_values = processed_dustmaps['EBV'].values
                ebv_values = np.nan_to_num(ebv_values, nan=0.0)
                df.loc[valid_coords, 'EBV'] = ebv_values
            
            return df
            
        except Exception as e:
            print(f"消光查询错误: {e}")
            df['EBV'] = 0.0
            return df

    def _add_magnitudes_to_df(self, df):
        """为DataFrame添加星等列（向量化）"""
        try:
            if 'distance_to_sun_pc' not in df.columns:
                return df
            
            distance_pc = df['distance_to_sun_pc'].values
            ebv = df['EBV'].values
            bands = ['u', 'g', 'r', 'i', 'z', 'y']
            
            for band in bands:
                abs_mag_col = f'{band}_mag'
                obs_mag_col = f'{band}_mag_observed'
                
                if abs_mag_col not in df.columns:
                    continue
                
                absolute_mag = df[abs_mag_col].values
                observed_mag = np.full_like(absolute_mag, np.nan)
                
                # 有效掩码
                valid = ((distance_pc > 0) & 
                        ~np.isnan(distance_pc) & 
                        ~np.isinf(distance_pc) & 
                        ~np.isnan(absolute_mag))
                
                if np.any(valid):
                    # 距离模数
                    dm = 5 * np.log10(distance_pc[valid]) - 5
                    # 消光改正
                    extinction = self.extinction_coefficients[band] * ebv[valid]
                    observed_mag[valid] = absolute_mag[valid] + dm + extinction
                
                df[obs_mag_col] = observed_mag
            
            return df
            
        except Exception as e:
            print(f"星等计算错误: {e}")
            return df

    def process_single_component(self, input_file):
        """处理单个文件（一次性加载，批量处理）"""
        try:
            if not os.path.exists(input_file):
                print(f"文件不存在: {input_file}")
                return False
            
            # 一次性读取整个文件（如果内存足够）
            print(f"读取文件: {os.path.basename(input_file)}")
            df = pd.read_hdf(input_file, key='conv')
            nrows = len(df)
            print(f"  总行数: {nrows:,}")
            
            # 步骤1: 添加坐标列
            print("  添加坐标列...")
            df = self._add_coordinates_to_df(df)
            
            # 步骤2: 添加消光值列（一次性查询）
            print("  查询消光值...")
            df = self._add_extinction_to_df(df)
            
            # 步骤3: 添加星等列
            print("  计算观测星等...")
            df = self._add_magnitudes_to_df(df)
            
            # 保存回文件（用临时文件）
            print("  保存文件...")
            temp_file = input_file + ".tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            df.to_hdf(temp_file, key='conv', mode='w', format='table', 
                     data_columns=True, index=False)
            shutil.move(temp_file, input_file)
            
            print(f"处理完成: {input_file}")
            
            # 验证添加的列
            original_columns = ['x_pc', 'y_pc', 'z_pc'] + [f'{band}_mag' for band in ['u', 'g', 'r', 'i', 'z', 'y']]
            new_columns = [col for col in df.columns 
                         if col not in original_columns and 
                         (col in ['l', 'b', 'd', 'distance_to_sun_pc', 'EBV'] 
                          or col.endswith('_mag_observed'))]
            print(f"  成功添加 {len(new_columns)} 个新列")
            
            # 释放内存
            del df
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return False

    def process_all_components(self, base_dir, components=None):
        """处理所有银河系组分"""
        if components is None:
            components = ['0.1', '0.3', '1', '3']
        
        print(f"开始处理 {len(components)} 个组分")
        success_count = 0
        
        for comp in tqdm(components, desc="总体进度"):
            print(f"\n{'='*60}")
            print(f"处理组分: {comp}")
            print(f"{'='*60}")
            
            # 输入文件
            input_file = f"{base_dir}/dwd_{comp}_interpolated.h5"
            
            if not os.path.exists(input_file):
                print(f"文件不存在: {input_file}")
                # 尝试查找其他格式
                alt_file = f"{base_dir}/dwd_{comp}.h5"
                if os.path.exists(alt_file):
                    print(f"找到备用文件: {alt_file}")
                    input_file = alt_file
                else:
                    continue
            
            success = self.process_single_component(input_file)
            if success:
                success_count += 1
                print(f"{comp} 处理成功")
            else:
                print(f"{comp} 处理失败")
        
        print(f"\n处理完成: {success_count}/{len(components)} 个组分成功")
        return success_count

if __name__ == "__main__":
    # 初始化处理器
    processor = SingleBatchProgressProcessor(n_cores=26)
    
    # 处理所有组件
    base_directory = "/publicfs10/fs10-m9/home/m9s003101/cosmic/LISA范围之内的DWD"
    processor.process_all_components(base_directory)

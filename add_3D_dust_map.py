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

    def _add_coordinates_to_batch(self, batch_df):
        """为数据批次添加坐标列"""
        try:
            # 向量化坐标转换
            x_pc = batch_df['x_pc'].values.astype(float)
            y_pc = batch_df['y_pc'].values.astype(float)
            z_pc = batch_df['z_pc'].values.astype(float)
            
            # 转换为kpc
            x_kpc = (x_pc * u.pc).to(u.kpc)
            y_kpc = (y_pc * u.pc).to(u.kpc)
            z_kpc = (z_pc * u.pc).to(u.kpc)
            
            # 批量坐标转换
            galactocentric_coords = coord.Galactocentric(
                x=x_kpc, y=y_kpc, z=z_kpc,
                galcen_distance=8.5*u.kpc, z_sun=0*u.pc
            )
            
            try:
                galactic_coords = galactocentric_coords.transform_to(coord.Galactic())
            except Exception:
                icrs_coords = galactocentric_coords.transform_to(coord.ICRS())
                galactic_coords = icrs_coords.transform_to(coord.Galactic())
            
            # 添加新列到原DataFrame
            batch_df['l'] = galactic_coords.l.deg
            batch_df['b'] = galactic_coords.b.deg
            batch_df['distance_to_sun_pc'] = galactic_coords.distance.to(u.pc).value
            batch_df['d'] = batch_df['distance_to_sun_pc'] / 1000.0
            
            # 验证距离是否有效
            invalid_distance = (batch_df['distance_to_sun_pc'] <= 0) | np.isnan(batch_df['distance_to_sun_pc'])
            if np.any(invalid_distance):
                batch_df.loc[invalid_distance, 'distance_to_sun_pc'] = np.nan
                batch_df.loc[invalid_distance, 'd'] = np.nan
            
            return batch_df
            
        except Exception as e:
            print(f"坐标转换错误: {e}")
            return batch_df

    def _add_extinction_to_batch(self, batch_df):
        """为数据批次添加消光值列（完全禁用内部输出）"""
        try:
            # 检查是否已有坐标列
            if 'l' not in batch_df.columns or 'b' not in batch_df.columns or 'd' not in batch_df.columns:
                batch_df['EBV'] = 0.0
                return batch_df
            
            # 检查是否有有效坐标（非空、非nan）
            valid_coords = (~batch_df['l'].isna()) & (~batch_df['b'].isna()) & (~batch_df['d'].isna())
            
            # 初始化消光列为0
            batch_df['EBV'] = 0.0
            
            if np.any(valid_coords):
                # 只处理有效坐标的行
                dustmaps_input = batch_df.loc[valid_coords, ['l', 'b', 'd']].copy()
                dustmaps_input = dustmaps_input.reset_index(drop=True)
                
                # 查询消光值
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    processed_dustmaps = dustmaps3d_from_df(
                        dustmaps_input, 
                        n_process=self.n_cores,
                        chunk_size=len(batch_df)
                    )
                
                # 将消光值填入对应的行
                batch_df.loc[valid_coords, 'EBV'] = processed_dustmaps['EBV'].values
            
            return batch_df
            
        except Exception as e:
            print(f"消光查询错误: {e}")
            batch_df['EBV'] = 0.0
            return batch_df

    def _add_magnitudes_to_batch(self, batch_df):
        """为数据批次添加星等列"""
        try:
            # 检查是否已有必要列
            if 'distance_to_sun_pc' not in batch_df.columns:
                return batch_df
            
            # 向量化星等计算
            distance_pc = batch_df['distance_to_sun_pc'].values
            ebv = batch_df['EBV'].values
            
            bands = ['u', 'g', 'r', 'i', 'z', 'y']
            
            for band in bands:
                abs_mag_col = f'{band}_mag'
                observed_mag_col = f'{band}_mag_observed'
                
                if abs_mag_col not in batch_df.columns:
                    continue
                
                absolute_mag = batch_df[abs_mag_col].values
                observed_mag = np.full_like(absolute_mag, np.nan)
                
                # 向量化计算：距离有效且绝对星等有效
                valid_mask = ((distance_pc > 0) & 
                            ~np.isnan(distance_pc) & 
                            ~np.isinf(distance_pc) & 
                            ~np.isnan(absolute_mag))
                
                if np.any(valid_mask):
                    extinction_correction = self.extinction_coefficients[band] * ebv[valid_mask]
                    observed_mag[valid_mask] = (absolute_mag[valid_mask] + 
                                               5 * np.log10(distance_pc[valid_mask]) - 5 +
                                               extinction_correction)
                
                # 直接添加观测星等列到原DataFrame
                batch_df[observed_mag_col] = observed_mag
            
            return batch_df
            
        except Exception as e:
            print(f"星等计算错误: {e}")
            return batch_df

    def process_single_component(self, input_file):
        """使用单一批次进度条处理单个银河系组分"""
        try:
            if not os.path.exists(input_file):
                print(f"输入文件不存在: {input_file}")
                return False
            
            # 获取总行数
            with pd.HDFStore(input_file, 'r') as store:
                nrows = store.get_storer('conv').nrows
            
            print(f"处理文件: {os.path.basename(input_file)}")
            print(f"数据总量: {nrows:,} 行")
            
            # 设置批次大小
            batch_size = 20000
            total_batches = (nrows + batch_size - 1) // batch_size
            
            # 创建单一进度条（按批次更新）
            pbar = tqdm(total=total_batches, desc=f"处理批次", 
                       unit='batch', mininterval=0.5, maxinterval=1.0)
            
            # 创建临时文件用于安全更新
            temp_file = input_file + ".tmp"
            
            # 如果临时文件存在，删除它
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            first_batch = True
            
            for batch_idx in range(total_batches):
                start = batch_idx * batch_size
                stop = min(start + batch_size, nrows)
                
                # 读取当前批次数据
                with pd.HDFStore(input_file, 'r') as store:
                    batch_df = store.select('conv', start=start, stop=stop)
                
                if batch_df.empty:
                    pbar.update(1)
                    continue
                
                # 步骤1: 添加坐标列
                batch_df = self._add_coordinates_to_batch(batch_df)
                
                # 步骤2: 添加消光值列
                batch_df = self._add_extinction_to_batch(batch_df)
                
                # 步骤3: 添加星等列
                batch_df = self._add_magnitudes_to_batch(batch_df)
                
                # 将更新后的批次写入临时文件
                with pd.HDFStore(temp_file, mode='a') as store:
                    if first_batch:
                        store.put('conv', batch_df, format='table', 
                                data_columns=True, append=False)
                        first_batch = False
                    else:
                        store.append('conv', batch_df, format='table')
                
                # 更新批次进度条
                pbar.update(1)
                pbar.set_postfix({
                    '当前批次': f'{batch_idx+1}/{total_batches}',
                    '已处理行数': f'{(batch_idx+1)*batch_size:,}'
                })
            
            # 关闭进度条
            pbar.close()
            
            # 用临时文件替换原文件
            import shutil
            shutil.move(temp_file, input_file)
            
            print(f"处理完成: {input_file}")
            
            # 验证添加的列
            with pd.HDFStore(input_file, 'r') as store:
                final_df = store.select('conv', start=0, stop=1)
                original_columns = ['x_pc', 'y_pc', 'z_pc'] + [f'{band}_mag' for band in ['u', 'g', 'r', 'i', 'z', 'y']]
                new_columns = [col for col in final_df.columns 
                             if col not in original_columns and 
                             (col in ['l', 'b', 'd', 'distance_to_sun_pc', 'EBV'] 
                              or col.endswith('_mag_observed'))]
                print(f"成功添加 {len(new_columns)} 个新列")
            
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
            components = ['halo', 'thick_disc', 'bulge', 'thin_disc']
        
        print(f"开始处理 {len(components)} 个银河系组分")
        success_count = 0
        
        for comp in components:
            print(f"\n{'='*60}")
            print(f"处理组分: {comp}")
            print(f"{'='*60}")
            
            input_file = f"{base_dir}/dwd_{comp}_interpolated.h5"
            
            if not os.path.exists(input_file):
                print(f"文件不存在: {input_file}")
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
    base_directory = "different_galactic_component_dwd_population_detached"
    processor.process_all_components(base_directory)

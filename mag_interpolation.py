#!/usr/bin/env python3
"""
作者：zhao
日期：2025年
python mag_interpolation.py
DWD冷却序列处理脚本
功能：
读取银河系各组分DWD数据，判断系统是否在冷却序列范围内，并进行插值计算
使用自然邻域法 + Delaunay三角剖分进行插值，可以自由选取
插值得到的参数：双星温度以及LSST6个绝对星等
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import ConvexHull, Delaunay
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import gc

# ==========================================================
# 自然邻域插值器
# ==========================================================
class NaturalNeighborInterpolator:
    """自然邻域插值器，基于Sibson的自然邻域法"""
    def __init__(self, points, values):
        self.points = np.asarray(points)
        self.values = np.asarray(values)
        self.delaunay = Delaunay(self.points)
        
    def __call__(self, query_point):
        query_point = np.asarray(query_point)
        
        # 检查点是否在凸包内
        simplex_index = self.delaunay.find_simplex([query_point])
        if simplex_index[0] == -1:
            return np.nan
        
        # 获取查询点的自然邻点
        simplex = self.delaunay.simplices[simplex_index[0]]
        neighbor_indices = simplex
        
        # 计算权重
        weights = []
        neighbor_values = []
        
        for idx in neighbor_indices:
            dist = np.linalg.norm(self.points[idx] - query_point)
            if dist < 1e-10:
                return self.values[idx]
            
            weight = 1.0 / dist
            weights.append(weight)
            neighbor_values.append(self.values[idx])
        
        weights = np.array(weights)
        neighbor_values = np.array(neighbor_values)
        
        if weights.sum() > 0:
            weights /= weights.sum()
            return np.dot(weights, neighbor_values)
        else:
            return np.nan

# ==========================================================
# Cooling track 插值器
# ==========================================================
class CoolingTrackInterpolator:
    def __init__(self, track_dir, min_mass=0.2, max_mass=1.3, verbose=False):
        self.track_dir = track_dir
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.verbose = verbose
        self.cooling_data = None
        self.delaunay_interpolators = {}
        self.natural_interpolators = {}
        self.age_max_interp = None
        self.age_min_interp = None
        self.load_cooling_tracks()

    def load_cooling_tracks(self):
        """加载蒙特利尔冷却表格文件（Table_Mass_*_with_LSST 格式）"""
        dfs = []
        
        files = sorted([f for f in os.listdir(self.track_dir) if f.startswith("Table_Mass_") and "_with_LSST" in f])
        
        if not files:
            files = sorted([f for f in os.listdir(self.track_dir) if f.startswith("Table_Mass_")])
            if self.verbose:
                print(f"未找到带 LSST 星等的文件，尝试读取普通表格文件，共 {len(files)} 个文件...")
        else:
            if self.verbose:
                print(f"正在加载蒙特利尔冷却表格文件（含 LSST 星等），共 {len(files)} 个文件...")
        
        for f in files:
            try:
                mass_str = f.replace("Table_Mass_", "").replace("_with_LSST", "")
                mass = float(mass_str)
                fpath = os.path.join(self.track_dir, f)
                
                with open(fpath, 'r') as file:
                    lines = file.readlines()
                
                data_start = 2
                data_lines = []
                for i in range(data_start, len(lines)):
                    line = lines[i].strip()
                    if line and not line.startswith('#'):
                        data_lines.append(line)
                
                if not data_lines:
                    continue
                
                rows = []
                for line in data_lines:
                    parts = line.split()
                    if len(parts) < 46:
                        continue
                    
                    try:
                        teff = float(parts[0])
                        logg = float(parts[1])
                        mbol = float(parts[2])
                        
                        n_cols = len(parts)
                        u_mag = float(parts[-6]) if n_cols >= 6 else np.nan
                        g_mag = float(parts[-5]) if n_cols >= 5 else np.nan
                        r_mag = float(parts[-4]) if n_cols >= 4 else np.nan
                        i_mag = float(parts[-3]) if n_cols >= 3 else np.nan
                        z_mag = float(parts[-2]) if n_cols >= 2 else np.nan
                        y_mag = float(parts[-1]) if n_cols >= 1 else np.nan
                        
                        age = float(parts[-7]) if n_cols >= 7 else np.nan
                        
                        logTeff = np.log10(teff)
                        Mbol_sun = 4.74
                        logL = -0.4 * (mbol - Mbol_sun)
                        
                        L_sun = 3.828e33
                        sigma = 5.670374419e-5
                        Rsun_cm = 6.957e10
                        L_ratio = 10 ** (-0.4 * (mbol - Mbol_sun))
                        L = L_sun * L_ratio
                        R_cm = np.sqrt(L / (4 * np.pi * sigma * teff**4))
                        R_Rsun = R_cm / Rsun_cm
                        
                        rows.append({
                            'M_WD': mass,
                            'logTeff': logTeff,
                            'logL': logL,
                            'logg': logg,
                            'age_Gyr': age / 1e9,
                            'R_Rsun': R_Rsun,
                            'u_mag': u_mag,
                            'g_mag': g_mag,
                            'r_mag': r_mag,
                            'i_mag': i_mag,
                            'z_mag': z_mag,
                            'y_mag': y_mag,
                        })
                    except (ValueError, IndexError):
                        continue
                
                if rows:
                    dfs.append(pd.DataFrame(rows))
                    
            except Exception as e:
                if self.verbose:
                    print(f"  读取文件 {f} 失败: {e}")
                continue

        if not dfs:
            raise RuntimeError(f"没有成功读取任何冷却表格文件，目录: {self.track_dir}")
            
        self.cooling_data = pd.concat(dfs, ignore_index=True)
        
        if self.verbose:
            print(f"总数据点: {len(self.cooling_data)}")
            print(f"质量范围: {self.cooling_data['M_WD'].min():.2f} - {self.cooling_data['M_WD'].max():.2f} M☉")
        
        self._remove_duplicate_age_points()
        self._setup_interpolators()

    def _remove_duplicate_age_points(self):
        """处理重复的冷却时间点"""
        if self.verbose:
            print("正在处理重复的冷却时间点...")
        
        original_count = len(self.cooling_data)
        self.cooling_data = self.cooling_data.sort_values(['M_WD', 'age_Gyr', 'logTeff'])
        self.cooling_data = self.cooling_data.drop_duplicates(subset=['M_WD', 'age_Gyr'], keep='first')
        
        final_count = len(self.cooling_data)
        removed_count = original_count - final_count
        
        if self.verbose and removed_count > 0:
            print(f"已移除 {removed_count} 个重复数据点，保留 {final_count} 个唯一数据点")

    def _setup_interpolators(self):
        """设置插值器"""
        points_2d = self.cooling_data[['M_WD', 'age_Gyr']].values
        target_columns = ['logTeff', 'R_Rsun', 'u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag', 'logL']
        
        for col in target_columns:
            values = self.cooling_data[col].values
            self.delaunay_interpolators[col] = LinearNDInterpolator(
                points_2d, values, fill_value=np.nan, rescale=True
            )
            self.natural_interpolators[col] = NaturalNeighborInterpolator(points_2d, values)
        
        self._setup_boundary_interpolators()
        
        if self.verbose:
            print("插值器初始化完成")

    def _setup_boundary_interpolators(self):
        """设置边界插值器"""
        age_max_by_mass = self.cooling_data.groupby("M_WD")["age_Gyr"].max().sort_index()
        self.age_max_interp = interp1d(
            age_max_by_mass.index.values, age_max_by_mass.values,
            bounds_error=False, fill_value=(age_max_by_mass.values[0], age_max_by_mass.values[-1])
        )
        
        age_min_by_mass = self.cooling_data.groupby("M_WD")["age_Gyr"].min().sort_index()
        self.age_min_interp = interp1d(
            age_min_by_mass.index.values, age_min_by_mass.values,
            bounds_error=False, fill_value=(age_min_by_mass.values[0], age_min_by_mass.values[-1])
        )

    def interpolate_star(self, mass, age, method="natural"):
        """插值单颗星的参数"""
        mass_used = np.clip(mass, self.min_mass, self.max_mass)
        age_max = float(self.age_max_interp(mass_used))
        age_min = float(self.age_min_interp(mass_used))
        age_used = np.clip(age, age_min, age_max)
        
        out = {}
        
        try:
            if method == "natural":
                point = np.array([mass_used, age_used])
                for col in self.natural_interpolators:
                    result = self.natural_interpolators[col](point)
                    out[col] = float(result) if not np.isnan(result) else np.nan
            elif method == "delaunay":
                query_array = np.array([[mass_used, age_used]])
                for col in self.delaunay_interpolators:
                    result = self.delaunay_interpolators[col](query_array)
                    out[col] = float(result[0]) if not np.isnan(result[0]) else np.nan
            else:
                raise ValueError(f"不支持的插值方法: {method}")
        except Exception:
            point = np.array([mass_used, age_used])
            for col in self.natural_interpolators:
                result = self.natural_interpolators[col](point)
                out[col] = float(result) if not np.isnan(result) else np.nan
        
        return out

    # ==========================================================
    # 处理双白矮星 DataFrame - 只保留前19列原始数据
    # ==========================================================
    def process_dwd_dataframe(self, df, current_universe_age=13700, method="natural"):
        """
        处理双白矮星数据框
        只保留原始数据的前19列 + 新计算的列
        """
        out_rows = []
        
        # 获取原始列名
        all_cols = df.columns.tolist()
        # 保留前19列
        keep_cols = all_cols[:19] if len(all_cols) >= 19 else all_cols
        
        for row in df.itertuples(index=False):
            # 只保留前19列的数据
            row_out = {col: getattr(row, col) for col in keep_cols}
            mags_1, mags_2 = {}, {}

            for idx in [1, 2]:
                m = getattr(row, f'mass_{idx}')
                tphys = getattr(row, 'tphys')
                tbirth = getattr(row, 'tbirth')
                aj = getattr(row, f'aj_{idx}')
                t = (current_universe_age - (tphys + tbirth) + aj) / 1000
                R_cosmic = getattr(row, f'rad_{idx}')

                m_interp = np.clip(m, self.min_mass, self.max_mass)
                i = self.interpolate_star(m_interp, t, method)

                if m < self.min_mass or m > self.max_mass:
                    R_ref = i['R_Rsun']
                    if np.isnan(R_ref) or R_ref <= 0:
                        logT = np.nan
                        mags = {b: np.nan for b in ['u','g','r','i','z','y']}
                    else:
                        dm = -5.0 * np.log10(R_cosmic / R_ref)
                        T_ref = 10**i['logTeff']
                        logT = np.log10(T_ref * np.sqrt(R_ref / R_cosmic))
                        mags = {b: i[f'{b}_mag'] + dm for b in ['u','g','r','i','z','y']}
                else:
                    logT = i['logTeff']
                    mags = {b: i[f'{b}_mag'] for b in ['u','g','r','i','z','y']}

                row_out[f'logTeff_{idx}'] = logT

                if idx == 1:
                    mags_1 = mags
                else:
                    mags_2 = mags

            # 双星合成星等
            for b in ['u','g','r','i','z','y']:
                if np.isnan(mags_1.get(b, np.nan)) or np.isnan(mags_2.get(b, np.nan)):
                    row_out[f'{b}_mag'] = np.nan
                else:
                    f_tot = 10**(-0.4 * mags_1[b]) + 10**(-0.4 * mags_2[b])
                    if f_tot <= 0:
                        row_out[f'{b}_mag'] = np.nan
                    else:
                        row_out[f'{b}_mag'] = -2.5 * np.log10(f_tot)

            out_rows.append(row_out)

        return pd.DataFrame(out_rows)


# ==========================================================
# 并行 worker
# ==========================================================
def init_worker(track_dir, method):
    """工作进程初始化函数（静默模式）"""
    global global_interpolator
    global_interpolator = CoolingTrackInterpolator(track_dir, verbose=False)

def _worker(args):
    """工作函数"""
    df_chunk, current_universe_age, method = args
    try:
        return global_interpolator.process_dwd_dataframe(df_chunk, current_universe_age, method)
    except Exception as e:
        return pd.DataFrame()


# ==========================================================
# 主程序
# ==========================================================
def main():
    cooling_dir = "/home/zhao/cosmic/alpha=0.1/DWD/AllTables_with_LSST"
    base_dir = "different_galactic_component_dwd_population_detached"
    
    nproc = min(cpu_count(), 26)
    current_universe_age = 13700
    method = "natural"

    print(f"使用 {nproc} 个进程进行并行处理")
    print(f"冷却序列目录: {cooling_dir}")

    components = ['halo','thick_disc','bulge','thin_disc']
    
    with tqdm(total=len(components), desc="总体进度", position=0) as main_pbar:
        for comp in components:
            infile = f"{base_dir}/dwd_3e-5_high_SNR_alpha=3_with_inclination.h5"
            if not os.path.exists(infile):
                print(f"文件不存在: {infile}")
                main_pbar.update(1)
                continue

            try:
                with pd.HDFStore(infile, 'r') as store:
                    nrows = store.get_storer('conv').nrows
                
                chunk_size = min(20000, max(1000, nrows // (nproc * 2)))
                total_chunks = (nrows + chunk_size - 1) // chunk_size
                
                with tqdm(total=total_chunks, desc=f"处理 {comp}", position=1, leave=False) as comp_pbar:
                    outdir = os.path.dirname(infile)
                    outfile = f"{outdir}/dwd_{comp}_interpolated.h5"
                    
                    if os.path.exists(outfile):
                        os.remove(outfile)

                    with Pool(processes=nproc, initializer=init_worker, 
                             initargs=(cooling_dir, method)) as pool:
                        
                        first_chunk = True
                        for chunk_id in range(total_chunks):
                            start = chunk_id * chunk_size
                            stop = min(start + chunk_size, nrows)
                            
                            df_chunk = pd.read_hdf(infile, key='conv', start=start, stop=stop)
                            if len(df_chunk) == 0:
                                comp_pbar.update(1)
                                continue
                            
                            sub_chunk_size = max(100, len(df_chunk) // nproc)
                            sub_chunks = []
                            for i in range(0, len(df_chunk), sub_chunk_size):
                                sub_chunk = df_chunk.iloc[i:i + sub_chunk_size]
                                sub_chunks.append((sub_chunk, current_universe_age, method))
                            
                            results = []
                            for result in pool.imap(_worker, sub_chunks):
                                results.append(result)
                            
                            if results:
                                df_out = pd.concat(results, ignore_index=True)
                                
                                mode = 'a' if not first_chunk else 'w'
                                df_out.to_hdf(outfile, key='conv', mode=mode, 
                                            format='table', append=not first_chunk, index=False)
                                first_chunk = False
                            
                            comp_pbar.update(1)
                            
                            del df_chunk, sub_chunks, results, df_out
                            gc.collect()

                    comp_pbar.refresh()
                
                main_pbar.update(1)
                main_pbar.set_postfix(当前组分=comp, 总行数=nrows)
                
            finally:
                gc.collect()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

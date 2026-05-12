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
from scipy.spatial import Delaunay
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import gc

# ==========================================================
# 物理常数
# ==========================================================
M_Ch = 1.44
M_p = 5.7e-4

def radius_from_mass(mass):
    """Verbunt & Rappaport (1988) 质量-半径关系"""
    if mass <= 0 or mass >= M_Ch:
        return 0.008
    x = mass / M_Ch
    term1 = x**(-2/3) - x**(2/3)
    if term1 <= 0:
        return 0.008
    y = mass / M_p
    term2 = 1 + 3.5 * y**(-2/3) + y**(-1)
    return 0.0114 * np.sqrt(term1) * term2**(-2/3)


# ==========================================================
# Cooling track 插值器（支持 LSST 和 CSST）
# ==========================================================
class CoolingTrackInterpolator:
    def __init__(self, track_dir, min_mass=0.2, max_mass=1.3, survey='LSST', verbose=False):
        """
        survey: 'LSST' 或 'CSST'
        """
        self.track_dir = track_dir
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.survey = survey
        self.verbose = verbose
        
        self.points = None
        self.interpolators = {}
        self.delaunay = None
        self.age_max_interp = None
        self.age_min_interp = None
        
        self._load_data()
        self._build_interpolators()

    def _load_data(self):
        """加载冷却数据，并计算半径"""
        dfs = []
        
        if not os.path.exists(self.track_dir):
            raise FileNotFoundError(f"目录不存在: {self.track_dir}")
        
        files = sorted([f for f in os.listdir(self.track_dir) if f.startswith("Table_Mass_")])
        
        for f in files:
            try:
                mass = float(f.replace("Table_Mass_", "").replace("_with_LSST", "").replace("_with_CSST", ""))
                fpath = os.path.join(self.track_dir, f)
                
                with open(fpath, 'r') as file:
                    lines = file.readlines()
                
                data_lines = [l.strip() for l in lines[2:] if l.strip() and not l.startswith('#')]
                
                for line in data_lines:
                    parts = line.split()
                    # 根据 survey 决定需要多少列
                    if self.survey == 'LSST':
                        # 预期列: Teff, logg, Mbol, BC, ... 然后 LSST 6 个波段 (u,g,r,i,z,y)
                        # 从表格中解析：需要知道位置，假设 LSST 表格中 u,g,r,i,z,y 是第 27-32 列（索引从0开始）
                        # 实际要根据文件格式调整，这里基于原代码的假设
                        if len(parts) < 46:  # LSST 表格有 46 列以上
                            continue
                        teff = float(parts[0])
                        mbol = float(parts[2])
                        age = float(parts[-7]) / 1e9
                        u_mag = float(parts[-6])
                        g_mag = float(parts[-5])
                        r_mag = float(parts[-4])
                        i_mag = float(parts[-3])
                        z_mag = float(parts[-2])
                        y_mag = float(parts[-1])
                        
                        row = {
                            'M_WD': mass,
                            'age_Gyr': age,
                            'logTeff': np.log10(teff),
                            'R_Rsun': radius_from_mass(mass),
                            'u_mag': u_mag,
                            'g_mag': g_mag,
                            'r_mag': r_mag,
                            'i_mag': i_mag,
                            'z_mag': z_mag,
                            'y_mag': y_mag,
                        }
                    else:  # CSST
                        # 假设 CSST 表格有 7 个波段：nuv,u,g,r,i,z,y，位于最后 7 列（顺序可能不同）
                        # 需要根据实际文件格式调整。这里假设最后 7 列是 nuv,u,g,r,i,z,y
                        if len(parts) < 7:
                            continue
                        teff = float(parts[0])
                        mbol = float(parts[2])
                        age = float(parts[-8]) / 1e9  # 假设年龄在倒数第7列（与LSST相同位置）
                        # 最后7列依次为 nuv, u, g, r, i, z, y
                        nuv_mag = float(parts[-7])
                        u_mag = float(parts[-6])
                        g_mag = float(parts[-5])
                        r_mag = float(parts[-4])
                        i_mag = float(parts[-3])
                        z_mag = float(parts[-2])
                        y_mag = float(parts[-1])
                        
                        row = {
                            'M_WD': mass,
                            'age_Gyr': age,
                            'logTeff': np.log10(teff),
                            'R_Rsun': radius_from_mass(mass),
                            'nuv_mag': nuv_mag,
                            'u_mag': u_mag,
                            'g_mag': g_mag,
                            'r_mag': r_mag,
                            'i_mag': i_mag,
                            'z_mag': z_mag,
                            'y_mag': y_mag,
                        }
                    
                    dfs.append(row)
            except Exception as e:
                if self.verbose:
                    print(f"解析文件 {f} 时出错: {e}")
                continue
        
        self.cooling_data = pd.DataFrame(dfs)
        if self.verbose:
            print(f"加载 {len(self.cooling_data)} 个数据点")

    def _build_interpolators(self):
        """构建插值器"""
        self.points = self.cooling_data[['M_WD', 'age_Gyr']].values
        self.delaunay = Delaunay(self.points)
        
        # 根据 survey 选择要插值的列
        if self.survey == 'LSST':
            cols = ['logTeff', 'R_Rsun', 'u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']
        else:
            cols = ['logTeff', 'R_Rsun', 'nuv_mag', 'u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']
        
        for col in cols:
            values = self.cooling_data[col].values
            self.interpolators[col] = LinearNDInterpolator(self.points, values, fill_value=np.nan)
        
        # 年龄边界
        age_max = self.cooling_data.groupby('M_WD')['age_Gyr'].max().sort_index()
        age_min = self.cooling_data.groupby('M_WD')['age_Gyr'].min().sort_index()
        self.age_max_interp = interp1d(age_max.index, age_max.values, bounds_error=False, fill_value=(age_max.values[0], age_max.values[-1]))
        self.age_min_interp = interp1d(age_min.index, age_min.values, bounds_error=False, fill_value=(age_min.values[0], age_min.values[-1]))

    def interpolate(self, mass, age, radius_cosmic=None):
        """
        插值单颗星
        返回: dict 或 None
        """
        # 质量边界
        if mass < self.min_mass:
            mass_used = self.min_mass
            use_scaling = True
        elif mass > self.max_mass:
            mass_used = self.max_mass
            use_scaling = True
        else:
            mass_used = mass
            use_scaling = False
        
        # 年龄边界
        age_max = float(self.age_max_interp(mass_used))
        age_min = float(self.age_min_interp(mass_used))
        age_used = np.clip(age, age_min, age_max)
        
        # 检查凸包
        if self.delaunay.find_simplex([[mass_used, age_used]])[0] == -1:
            return None
        
        # 插值
        point = np.array([mass_used, age_used])
        out = {}
        for col, interp in self.interpolators.items():
            val = interp(point)[0]
            out[col] = val if not np.isnan(val) else np.nan
        
        if np.isnan(out.get('logTeff', np.nan)):
            return None
        
        # 质量超出时缩放
        if use_scaling and radius_cosmic is not None and not np.isnan(radius_cosmic):
            R_ref = out.get('R_Rsun', np.nan)
            if not np.isnan(R_ref) and R_ref > 0:
                dm = -5.0 * np.log10(radius_cosmic / R_ref)
                for b in self._get_band_list():
                    mag_key = f'{b}_mag'
                    if mag_key in out and not np.isnan(out[mag_key]):
                        out[mag_key] += dm
        
        return out

    def _get_band_list(self):
        if self.survey == 'LSST':
            return ['u', 'g', 'r', 'i', 'z', 'y']
        else:
            return ['nuv', 'u', 'g', 'r', 'i', 'z', 'y']


# ==========================================================
# 处理双白矮星数据
# ==========================================================
def process_chunk(df, interp, current_age=13700):
    """处理单个数据块"""
    t = (current_age - (df['tphys'] + df['tbirth']).values) / 1000
    
    n = len(df)
    # 根据 survey 初始化 mags 字典
    bands = interp._get_band_list()
    mags = {b: np.full(n, np.nan) for b in bands}
    logTeff = {1: np.full(n, np.nan), 2: np.full(n, np.nan)}
    
    for idx in [1, 2]:
        mass_col = f'mass_{idx}'
        aj_col = f'aj_{idx}'
        rad_col = f'rad_{idx}'
        
        age_star = t + df[aj_col].values / 1000
        masses = df[mass_col].values
        radii = df[rad_col].values
        
        for i in range(n):
            if np.isnan(masses[i]) or np.isnan(age_star[i]):
                continue
            
            res = interp.interpolate(masses[i], age_star[i], radii[i])
            if res:
                logTeff[idx][i] = res.get('logTeff', np.nan)
                for b in bands:
                    mag_key = f'{b}_mag'
                    if mag_key not in res:
                        continue
                    if idx == 1:
                        mags[b][i] = res[mag_key]
                    else:
                        mag2 = res[mag_key]
                        mag1 = mags[b][i]
                        if not np.isnan(mag1) and not np.isnan(mag2):
                            flux = 10**(-0.4*mag1) + 10**(-0.4*mag2)
                            mags[b][i] = -2.5 * np.log10(flux)
                        elif not np.isnan(mag2):
                            mags[b][i] = mag2
    
    # 构建输出
    out = {col: df[col].values.copy() for col in df.columns[:19]}
    out['logTeff_1'] = logTeff[1]
    out['logTeff_2'] = logTeff[2]
    for b in bands:
        out[f'{b}_mag'] = mags[b]
    
    return pd.DataFrame(out)


# ==========================================================
# Worker
# ==========================================================
def init_worker(track_dir, survey):
    global _interpolator
    _interpolator = CoolingTrackInterpolator(track_dir, survey=survey, verbose=False)

def worker(args):
    df, current_age = args
    try:
        return process_chunk(df, _interpolator, current_age)
    except:
        return pd.DataFrame()


# ==========================================================
# 主程序
# ==========================================================
def main():
    # 选择使用 LSST 还是 CSST
    use_csst = True  # 改为 False 则使用 LSST
    
    if use_csst:
        cooling_dir = "/publicfs10/fs10-m9/home/m9s003101/cosmic/AllTables_with_CSST"  # CSST 冷却表格目录
        suffix = "_csst"
    else:
        cooling_dir = "/publicfs10/fs10-m9/home/m9s003101/cosmic/AllTables_with_LSST"
        suffix = "_lsst"
    
    #base_dir = "/publicfs10/fs10-m9/home/m9s003101/cosmic/LSST+LISA范围之内的DWD"
    base_dir = "/publicfs10/fs10-m9/home/m9s003101/cosmic/downsampled_merged_1pct"
    

    nproc = min(cpu_count(), 26)
    current_age = 13700

    print(f"使用 {nproc} 进程 | 冷却目录: {cooling_dir}")
    
    for comp in ['0.1', '0.3', '1', '3']:
        infile = f"{base_dir}/alpha={comp}_downsampled_10pct.h5"
        #infile = f"{base_dir}/dwd_1e-4_{comp}.h5"
        if not os.path.exists(infile):
            print(f"跳过: {infile}")
            continue
        
        with pd.HDFStore(infile, 'r') as store:
            nrows = store.get_storer('conv').nrows
        
        print(f"\n处理 alpha={comp}, 行数: {nrows:,}")
        
        chunk_size = min(50000, max(1000, nrows // (nproc * 2) or 1000))
        total = (nrows + chunk_size - 1) // chunk_size
        
        outfile = f"{base_dir}/dwd_{comp}_interpolated{suffix}.h5"
        if os.path.exists(outfile):
            os.remove(outfile)
        
        first = True
        with Pool(nproc, initializer=init_worker, initargs=(cooling_dir, 'CSST' if use_csst else 'LSST')) as pool:
            with tqdm(total=total, desc=f"alpha={comp}") as pbar:
                for i in range(total):
                    start = i * chunk_size
                    stop = min(start + chunk_size, nrows)
                    df = pd.read_hdf(infile, key='conv', start=start, stop=stop)
                    
                    if len(df) == 0:
                        pbar.update(1)
                        continue
                    
                    sub_size = max(100, len(df) // nproc)
                    tasks = [(df.iloc[j:j+sub_size], current_age) for j in range(0, len(df), sub_size)]
                    
                    results = []
                    for res in pool.imap(worker, tasks):
                        if len(res) > 0:
                            results.append(res)
                    
                    if results:
                        out = pd.concat(results, ignore_index=True)
                        mode = 'a' if not first else 'w'
                        out.to_hdf(outfile, key='conv', mode=mode, format='table', append=not first, index=False)
                        first = False
                    
                    pbar.update(1)
                    gc.collect()
        
        print(f"完成: {outfile}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

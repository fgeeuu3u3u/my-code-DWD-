#!/usr/bin/env python3
# 查询观测时间点以及误差（增加星等误差、观测星等和相位列）
#  python 查询观测时间点以及误差.py
#!/usr/bin/env python3
# 查询观测时间点以及误差（增加星等误差、观测星等和相位列）
# 修正：周期单位直接使用天，a=sep，倾角映射到0-90度

import os
import sys
import numpy as np
import pandas as pd
import ellc
from scipy.interpolate import interp1d
import rubin_sim.maf as maf
from rubin_sim.data import get_baseline
from astropy.coordinates import SkyCoord
import astropy.units as u
import re

# 导入冷却轨线插值器（确保路径正确）
from mag_interpolation_LSST_CSST import CoolingTrackInterpolator

# ================= 配置 =================
base_dir = "/home/zhao/cosmic/LISA范围中有光变曲线DWD"
track_dir = "/home/zhao/cosmic/alpha=0.1/DWD/AllTables_with_LSST"
interp = CoolingTrackInterpolator(track_dir, survey='LSST', verbose=False)

baseline_file = get_baseline()
print(f"Using baseline: {baseline_file}")

# LSST 误差参数
GAMMA = 0.039
SIGMA_SYS = 0.005

def get_r_mag(mass, age_Gyr, radius_Rsun=None):
    res = interp.interpolate(mass, age_Gyr, radius_Rsun)
    if res is None:
        return np.nan
    return res.get('r_mag', np.nan)

def compute_sbratio(mass1, age1, radius1, mass2, age2, radius2):
    mag1 = get_r_mag(mass1, age1, radius1)
    mag2 = get_r_mag(mass2, age2, radius2)
    if np.isnan(mag1) or np.isnan(mag2):
        return np.nan
    return 10.0 ** (-0.4 * (mag2 - mag1))

def lsst_sigma_mag(mag, m5):
    x = 10.0 ** (0.4 * (mag - m5))
    sigma_rand2 = (0.04 - GAMMA) * x + GAMMA * x * x
    sigma_rand = np.sqrt(max(sigma_rand2, 0.0))
    return np.sqrt(sigma_rand**2 + SIGMA_SYS**2)

def generate_mag_interpolator(rad1_norm, rad2_norm, sbratio, incl_deg, period_days,
                              r_mag_observed, a, n_phase=1000):
    """生成插值函数 mag(phase)，归一化使用相位0处的流量"""
    phase_dense = np.linspace(0, 1, n_phase)
    time_dense = phase_dense * period_days
    t_zero = 0.2 * period_days
    flux = ellc.lc(time_dense,
                   radius_1=rad1_norm, radius_2=rad2_norm,
                   sbratio=sbratio,
                   incl=incl_deg,
                   t_zero=t_zero,
                   period=period_days,
                   a=a,
                   shape_1='sphere', shape_2='sphere')
    # 使用第一个点（相位0）的流量作为归一化因子
    norm = flux[0]
    if norm <= 0:
        raise ValueError("归一化因子无效")
    flux_norm = flux / norm
    delta_mag = -2.5 * np.log10(flux_norm)
    mag_dense = r_mag_observed + delta_mag
    interp = interp1d(phase_dense, mag_dense, kind='linear',
                      bounds_error=False, fill_value=(mag_dense[-1], mag_dense[0]))
    return interp

def get_r_band_obs_bulk(ra_list, dec_list):
    """批量查询多个点的 r 波段观测，返回列表，每个元素为 (MJD, fiveSigmaDepth) 元组"""
    if not ra_list:
        return []
    slicer = maf.slicers.UserPointsSlicer(ra=ra_list, dec=dec_list)
    metric = maf.metrics.PassMetric(cols=['band', 'observationStartMJD', 'fiveSigmaDepth'])
    bundle = maf.MetricBundle(metric, slicer, constraint='', run_name='temp')
    bd = maf.metricBundles.make_bundles_dict_from_list([bundle])
    bg = maf.metricBundles.MetricBundleGroup(bd, baseline_file, out_dir='temp', results_db=None)
    bg.run_all()
    metric_vals = bundle.metric_values
    if isinstance(metric_vals, np.ma.MaskedArray):
        data_list = [metric_vals[i] for i in range(len(metric_vals))]
    else:
        data_list = [metric_vals] if not isinstance(metric_vals, (list, tuple)) else metric_vals
    results = []
    for data in data_list:
        if data is None or (hasattr(data, 'size') and data.size == 0):
            results.append((np.array([]), np.array([])))
            continue
        if not hasattr(data, 'dtype'):
            results.append((np.array([]), np.array([])))
            continue
        try:
            mask = data['band'] == 'r'
            obs_times = data['observationStartMJD'][mask]
            m5_vals = data['fiveSigmaDepth'][mask]
            results.append((obs_times, m5_vals))
        except Exception:
            results.append((np.array([]), np.array([])))
    return results

def process_top_folder(top):
    top_path = os.path.join(base_dir, top)
    h5_file = os.path.join(base_dir, top + ".h5")
    if not os.path.exists(h5_file):
        print(f"跳过 {top}: 缺少原始 HDF5 文件")
        return
    df = pd.read_hdf(h5_file, key='conv')
    df = df.reset_index(drop=True)
    source_dirs = [d for d in os.listdir(top_path) if d.startswith('source_')]
    source_dirs.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    print(f"处理 {top}: {len(source_dirs)} 个源")
    
    # 收集所有源的银河坐标
    ra_list, dec_list, period_list, src_list = [], [], [], []
    for src in source_dirs:
        idx = int(re.findall(r'\d+', src)[0])
        if idx >= len(df):
            continue
        row = df.iloc[idx]
        l, b = row['l'], row['b']
        period = row['porb']          # 单位：天
        c = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic')
        ra, dec = c.icrs.ra.deg, c.icrs.dec.deg
        ra_list.append(ra)
        dec_list.append(dec)
        period_list.append(period)
        src_list.append(src)
    if not src_list:
        return
    
    # 批量查询观测 (MJD, m5)
    obs_results = get_r_band_obs_bulk(ra_list, dec_list)
    
    # 对每个源，生成相位、理论星等和误差
    for i, (src, period_days, (obs_times, m5_vals)) in enumerate(zip(src_list, period_list, obs_results)):
        if len(obs_times) == 0:
            continue
        # 读取该源的行数据
        idx = int(re.findall(r'\d+', src)[0])
        row = df.iloc[idx]
        rad1 = row['rad_1']
        rad2 = row['rad_2']
        sep = row['sep']
        inc_rad = row['inclination']
        incl_deg = inc_rad * 180.0 / np.pi
        # 映射到 0-90 度
        if incl_deg > 90.0:
            incl_deg = 180.0 - incl_deg
        r_mag_observed = row['r_mag_observed']
        # 年龄计算
        current_age_Myr = 13700
        t = (current_age_Myr - (row['tphys'] + row['tbirth'])) / 1000.0
        age1 = t + row['aj_1'] / 1000.0
        age2 = t + row['aj_2'] / 1000.0
        sbratio = compute_sbratio(row['mass_1'], age1, rad1, row['mass_2'], age2, rad2)
        if np.isnan(sbratio):
            print(f"跳过 {src}: 表面亮度比无效")
            continue
        r1_norm = rad1 / sep
        r2_norm = rad2 / sep
        
        # 计算观测相位（固定参考历元）
        MJD_REF = 57000.0
        phases = ((obs_times - MJD_REF) / period_days) % 1.0
        
        # 生成理论光变曲线插值器（a 使用 sep，与生成光变曲线一致）
        try:
            interp_mag = generate_mag_interpolator(r1_norm, r2_norm, sbratio, incl_deg,
                                                   period_days, r_mag_observed, a=sep)
        except Exception as e:
            print(f"跳过 {src}: ellc 失败 - {e}")
            continue
        
        # 得到理论星等（作为观测星等）
        mag_obs = interp_mag(phases)
        # 计算星等误差
        sigma_mag = np.array([lsst_sigma_mag(m, m5) for m, m5 in zip(mag_obs, m5_vals)])
        
        # 写入文件：MJD, fiveSigmaDepth, sigma_mag, mag_obs, phase
        src_path = os.path.join(top_path, src)
        os.makedirs(src_path, exist_ok=True)
        out_file = os.path.join(src_path, "obs.csv")
        with open(out_file, 'w') as f:
            # 第一行写周期（天）供参考
            f.write(f"{period_days}\n")
            f.write("MJD,fiveSigmaDepth,sigma_mag,mag_obs,phase\n")
            for mjd, m5, sig, mag, ph in zip(obs_times, m5_vals, sigma_mag, mag_obs, phases):
                f.write(f"{mjd},{m5},{sig:.6f},{mag:.6f},{ph:.6f}\n")
        if i == 0:
            print(f"  已生成 {src} 的观测文件，包含 {len(obs_times)} 个点")

if __name__ == "__main__":
    top_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and 'lightcurve' in f]
    print(f"找到 {len(top_folders)} 个顶层文件夹")
    for top in top_folders:
        process_top_folder(top)
    print("所有观测文件生成完毕（已修正周期单位、a=sep、倾角映射）。")

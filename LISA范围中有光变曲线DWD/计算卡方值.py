



# python 计算卡方值.py





#!/usr/bin/env python3
# 计算卡方值.py（适配五列 obs.csv）

import os
import numpy as np
import pandas as pd
import re

# ================= 配置 =================
base_dir = "/home/zhao/cosmic/LISA范围中有光变曲线DWD"   # 顶层文件夹所在目录
output_dir = base_dir  # 结果文件保存在同一目录
n_mc = 100   # 蒙特卡洛次数

# LSST 测光误差模型参数（r 波段）
gamma = 0.039
sigma_sys = 0.005

def lsst_error(mag, m5):
    x = 10 ** (0.4 * (mag - m5))
    sigma_rand_sq = (0.04 - gamma) * x + gamma * x**2
    sigma_rand = np.sqrt(sigma_rand_sq)
    return np.sqrt(sigma_rand**2 + sigma_sys**2)

def read_obs_csv(obs_path):
    """
    读取 obs.csv（支持五列格式），返回 (period, mjd_array, m5_array)
    只使用前两列：MJD 和 fiveSigmaDepth
    """
    with open(obs_path, 'r') as f:
        lines = f.readlines()
    period = float(lines[0].strip())
    if len(lines) < 3:
        return period, np.array([]), np.array([])
    try:
        data = np.loadtxt(obs_path, skiprows=2, delimiter=',', usecols=(0,1))
    except:
        return period, np.array([]), np.array([])
    if data.size == 0:
        return period, np.array([]), np.array([])
    if data.ndim == 1:
        mjd = np.array([data[0]])
        m5 = np.array([data[1]])
    else:
        mjd = data[:, 0]
        m5 = data[:, 1]
    return period, mjd, m5

def read_lightcurve_csv(lc_path):
    df = pd.read_csv(lc_path)
    return df['phase'].values, df['mag'].values

def process_source(lc_path, obs_path, r_mag):
    phase_grid, mag_grid = read_lightcurve_csv(lc_path)
    period, obs_times, m5_vals = read_obs_csv(obs_path)
    if len(obs_times) == 0:
        return 0
    errors = np.array([lsst_error(r_mag, m5) for m5 in m5_vals])
    N = len(errors)
    count = 0
    for _ in range(n_mc):
        t0 = np.random.uniform(0, period)
        phases = ((obs_times - t0) / period) % 1.0
        theo = np.interp(phases, phase_grid, mag_grid)
        noise = np.random.normal(0, errors)
        obs_mag = theo + noise
        mean_mag = np.mean(obs_mag)
        chi2 = np.sum(((obs_mag - mean_mag) / errors)**2) / N
        if chi2 > 3.0:
            count += 1
    return count

def main():
    top_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and 'lightcurve' in f]
    print(f"找到 {len(top_folders)} 个顶层文件夹")
    
    for top in top_folders:
        top_path = os.path.join(base_dir, top)
        h5_file = os.path.join(base_dir, top + ".h5")
        if not os.path.exists(h5_file):
            print(f"跳过 {top}: 缺少原始 HDF5 文件")
            continue
        df_orig = pd.read_hdf(h5_file, key='conv')
        df_orig = df_orig.reset_index(drop=True)
        source_dirs = [d for d in os.listdir(top_path) if d.startswith('source_')]
        source_dirs.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        print(f"处理 {top}: {len(source_dirs)} 个源")
        
        results = []
        for src in source_dirs:
            idx = int(re.findall(r'\d+', src)[0])
            if idx >= len(df_orig):
                print(f"  跳过 {src}: 索引超出范围")
                continue
            lc_path = os.path.join(top_path, src, "lightcurve.csv")
            obs_path = os.path.join(top_path, src, "obs.csv")
            if not os.path.exists(lc_path) or not os.path.exists(obs_path):
                print(f"  跳过 {src}: 缺少 lightcurve.csv 或 obs.csv")
                continue
            r_mag = df_orig.iloc[idx]['r_mag_observed']
            count = process_source(lc_path, obs_path, r_mag)
            results.append({
                'source': src,
                'detection_count': count,
                'n_obs': len(pd.read_csv(obs_path, skiprows=1)['MJD']) if os.path.getsize(obs_path) > 0 else 0,
                'period': df_orig.iloc[idx]['porb'],
                'r_mag': r_mag
            })
            if len(results) % 100 == 0:
                print(f"  已处理 {len(results)} 个源")
        
        out_file = os.path.join(output_dir, f"chi2_results_{top}.csv")
        df_out = pd.DataFrame(results)
        df_out.to_csv(out_file, index=False)
        print(f"  结果已保存到 {out_file}")
        print(f"  该文件夹共 {len(results)} 个源，检测次数平均 {df_out['detection_count'].mean():.2f}, 中位数 {df_out['detection_count'].median():.2f}")
    
    print("所有处理完成。")

if __name__ == "__main__":
    main()
    

#!/usr/bin/env python3
"""
计算 LSST 对双白矮星的天体测量误差（位置和距离）
使用每个源的实际 r 波段观测次数（从 obs.csv 读取）。
对于 r < 21 的亮星，使用系统误差极限模型；
对于 r >= 21 使用表 3 插值。
蓝星修正：r >= 23 时视差误差乘 1.5。
输出 CSV 包含源索引、r_mag_observed、距离 d、n_obs，
以及 err_co_latitude_rad, err_longitude_rad, err_distance_relative。
python 位置+距离矩阵.py
"""



import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ================= 路径配置 =================
BASE_DIR = "/home/zhao/cosmic/LISA范围中有光变曲线DWD"
PASSED_DIR = os.path.join(BASE_DIR, "通过卡方值检验的DWD")
LIGHTCURVES_DIR = os.path.join(PASSED_DIR, "lightcurves")
OUTPUT_DIR = os.path.join(PASSED_DIR, "fisher_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LSST 表 3 数据（用于 r >= 21）
mag_table = np.array([21, 22, 23, 24])
sigma_xy_mas = np.array([11, 15, 31, 74])      # 单次曝光每坐标位置误差 (mas)
sigma_parallax_mas = np.array([0.6, 0.8, 1.3, 2.9])  # 10年视差精度 (mas)

# 亮星系统误差极限（r < 21）
BRIGHT_SIGMA_XY_MAS = 10.0        # 单次曝光位置误差 (mas)
BRIGHT_PARALLAX_MAS = 0.5         # 10年视差精度 (mas) 保守估计

# 插值函数（仅用于 r >= 21）
interp_xy = interp1d(mag_table, sigma_xy_mas, kind='linear', fill_value='extrapolate')
interp_parallax = interp1d(mag_table, sigma_parallax_mas, kind='linear', fill_value='extrapolate')

def mas_to_rad(mas):
    """毫角秒转弧度"""
    return mas * np.pi / (180 * 3600 * 1000)   # 1 mas = π/648000 rad

def get_obs_count(obs_csv_path):
    """读取 obs.csv，返回 r 波段观测点数（数据行数）"""
    if not os.path.exists(obs_csv_path):
        return 0
    try:
        df = pd.read_csv(obs_csv_path, skiprows=1)  # 第一行是周期，第二行是列名，数据从第三行开始
        return len(df)
    except:
        return 0

def compute_errors(r_mag, d_kpc, n_obs):
    """
    输入：r波段星等，距离 (kpc)，r波段观测次数
    返回：(err_co_latitude_rad, err_longitude_rad, err_distance_abs_kpc)
    """
    if r_mag < 21:
        # 亮星：使用系统误差极限模型
        sigma_xy = BRIGHT_SIGMA_XY_MAS        # mas
        sigma_parallax = BRIGHT_PARALLAX_MAS  # mas
    else:
        # 暗星：使用表 3 插值
        sigma_xy = interp_xy(r_mag)           # mas
        sigma_parallax = interp_parallax(r_mag)  # mas

    # 蓝星修正：对于 r >= 23，视差误差放大 1.5 倍（假设所有 DWD 为蓝星）
    if r_mag >= 23:
        sigma_parallax *= 1.5

    # 1. 位置误差（10年平均位置精度 = 单次曝光误差 / sqrt(n_obs)）
    if n_obs > 0:
        sigma_pos_rad = mas_to_rad(sigma_xy) / np.sqrt(n_obs)
    else:
        sigma_pos_rad = np.inf
    err_co_lat = sigma_pos_rad
    err_lon = sigma_pos_rad

    # 2. 距离绝对误差 (kpc): σ_d = d^2 * σ_π
    # 其中 d 单位 kpc，σ_π 单位 mas，由于 1 kpc 对应 1 mas 视差，公式正确
    err_d_abs = d_kpc * d_kpc * sigma_parallax   # 绝对误差 (kpc)

    return err_co_lat, err_lon, err_d_abs

def main():
    # 获取所有通过检验的 HDF5 文件
    h5_files = [f for f in os.listdir(PASSED_DIR) if f.endswith('.h5')]
    if not h5_files:
        print("未找到 HDF5 文件")
        sys.exit(1)

    for h5_file in h5_files:
        top = os.path.splitext(h5_file)[0]
        h5_path = os.path.join(PASSED_DIR, h5_file)
        df = pd.read_hdf(h5_path, key='conv')
        if 'r_mag_observed' not in df.columns or 'd' not in df.columns:
            print(f"跳过 {top}: 缺少 r_mag_observed 或 d 列")
            continue

        results = []
        for idx, row in df.iterrows():
            r_mag = row['r_mag_observed']
            d_kpc = row['d']
            obs_csv_path = os.path.join(LIGHTCURVES_DIR, top, f"source_{idx}", "obs.csv")
            n_obs = get_obs_count(obs_csv_path)
            if n_obs == 0:
                print(f"警告: 源 {idx} 在 {top} 中无观测数据，跳过")
                continue
            err_co_lat, err_lon, err_d_abs = compute_errors(r_mag, d_kpc, n_obs)
            results.append({
                'source_idx': idx,
                'r_mag_observed': r_mag,
                'distance_kpc': d_kpc,
                'n_obs': n_obs,
                'err_co_latitude_rad': err_co_lat,
                'err_longitude_rad': err_lon,
                'err_distance_abs_kpc': err_d_abs
            })
        if results:
            df_out = pd.DataFrame(results)
            out_csv = os.path.join(OUTPUT_DIR, f"lsst_astrometry_errors_{top}.csv")
            df_out.to_csv(out_csv, index=False)
            print(f"已保存 {len(results)} 个源的天体测量误差到 {out_csv}")
        else:
            print(f"未找到有效源 {top}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3




#  python 生成理论光变曲线.py



import os
import numpy as np
import pandas as pd
import ellc
import matplotlib
matplotlib.use('Agg')  # 无界面后端，避免显示图形
import matplotlib.pyplot as plt
from mag_interpolation_LSST_CSST import CoolingTrackInterpolator

# ================= 配置 =================
base_dir = "/home/zhao/cosmic/LISA范围中有光变曲线DWD"
track_dir = "/home/zhao/cosmic/alpha=0.1/DWD/AllTables_with_LSST"
interp = CoolingTrackInterpolator(track_dir, survey='LSST', verbose=False)

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
    return 10 ** (-0.4 * (mag2 - mag1))

def generate_lightcurve_for_row(row, output_dir, source_name):
    """为单个 DWD 生成光变曲线，保存 CSV 和图像"""
    rad1 = row['rad_1']
    rad2 = row['rad_2']
    sep = row['sep']
    inc_rad = row['inclination']
    incl_deg = inc_rad * 180.0 / np.pi
    period = row['porb']
    r_mag_observed = row['r_mag_observed']

    current_age_Myr = 13700
    t = (current_age_Myr - (row['tphys'] + row['tbirth'])) / 1000.0
    age1 = t + row['aj_1'] / 1000.0
    age2 = t + row['aj_2'] / 1000.0

    sbratio = compute_sbratio(row['mass_1'], age1, rad1, row['mass_2'], age2, rad2)
    if np.isnan(sbratio):
        print(f"  跳过 {source_name}: 表面亮度比无效")
        return

    r1_norm = rad1 / sep
    r2_norm = rad2 / sep
    phase = np.linspace(0, 1, 500)
    time = phase * period
    t_zero = 0.2 * period  # 可调整，将主掩食放在相位0.2附近

    try:
        flux = ellc.lc(time, radius_1=r1_norm, radius_2=r2_norm, sbratio=sbratio,
                       incl=incl_deg, t_zero=t_zero, period=period, a=sep,
                       shape_1='sphere', shape_2='sphere')
    except Exception as e:
        print(f"  错误: ellc 计算失败 for {source_name}: {e}")
        return

    norm = flux[0]   # 取第一个点的流量
    if norm <= 0:
        print(f"  警告: 归一化因子异常，跳过 {source_name}")
        return
    flux_norm = flux / norm
    delta_mag = -2.5 * np.log10(flux_norm)
    actual_mag = r_mag_observed + delta_mag

    # 保存 CSV
    csv_path = os.path.join(output_dir, "lightcurve.csv")
    pd.DataFrame({'phase': phase, 'mag': actual_mag}).to_csv(csv_path, index=False)

    # 保存图像
    plt.figure(figsize=(8,5))
    plt.plot(phase, actual_mag, 'b-')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Apparent r-band magnitude')
    plt.title(f'Light curve for {source_name}')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3)
    img_path = os.path.join(output_dir, "lightcurve.png")
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"  已生成 {csv_path} 和 {img_path}")

# ================= 主循环 =================
h5_files = [f for f in os.listdir(base_dir) if f.endswith('.h5')]
print(f"找到 {len(h5_files)} 个 HDF5 文件")

for h5_file in h5_files:
    file_path = os.path.join(base_dir, h5_file)
    base_name = os.path.splitext(h5_file)[0]
    output_file_dir = os.path.join(base_dir, base_name)
    os.makedirs(output_file_dir, exist_ok=True)
    print(f"\n处理文件: {h5_file} -> 输出目录: {output_file_dir}")

    try:
        df = pd.read_hdf(file_path, key='conv')
    except Exception as e:
        print(f"  读取失败: {e}")
        continue

    # 重置索引，使序号从 0 开始连续
    df = df.reset_index(drop=True)
    for idx, row in df.iterrows():
        source_name = f"source_{idx}"   # idx 从 0 开始
        out_subdir = os.path.join(output_file_dir, source_name)
        os.makedirs(out_subdir, exist_ok=True)
        print(f"  处理 {source_name} (共 {len(df)} 个)")
        generate_lightcurve_for_row(row, out_subdir, source_name)

print("\n所有光变曲线生成完毕！")

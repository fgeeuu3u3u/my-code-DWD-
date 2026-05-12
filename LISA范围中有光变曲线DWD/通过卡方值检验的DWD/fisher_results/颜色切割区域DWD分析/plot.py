#!/usr/bin/env python3
"""
绘制 α 颜色选择区域内 DWD 的 SNR vs 光变幅度 hexbin 密度图
并统计满足 SNR>4 且 amplitude>0.02 的源数量
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ellc
from mag_interpolation_LSST_CSST import CoolingTrackInterpolator

# ================= 配置 =================
alpha = "1"   # 可改为 "1" 或 "3"
base_dir = "/home/zhao/cosmic/LISA范围中有光变曲线DWD/通过卡方值检验的DWD/fisher_results/颜色切割区域DWD分析/LSST"

h5_file = os.path.join(base_dir, f"dwd_{alpha}_interpolated_color_selected.h5")
snr_file = os.path.join(base_dir, f"SigmasAE_LSST_{alpha}.dat")
track_dir = "/home/zhao/cosmic/alpha=0.1/DWD/AllTables_with_LSST"
interp = CoolingTrackInterpolator(track_dir, survey='LSST', verbose=False)

# ================= 辅助函数 =================
def get_r_mag(mass, age_Gyr, radius_Rsun=None):
    res = interp.interpolate(mass, age_Gyr, radius_Rsun)
    return res.get('r_mag', np.nan) if res is not None else np.nan

def compute_sbratio(mass1, age1, radius1, mass2, age2, radius2):
    mag1 = get_r_mag(mass1, age1, radius1)
    mag2 = get_r_mag(mass2, age2, radius2)
    if np.isnan(mag1) or np.isnan(mag2):
        return np.nan
    return 10.0 ** (-0.4 * (mag2 - mag1))

def normalize_inclination(inc_rad):
    deg = inc_rad * 180.0 / np.pi
    if deg > 90.0:
        deg = 180.0 - deg
    return deg

def compute_amplitude(row):
    rad1 = row['rad_1']
    rad2 = row['rad_2']
    sep = row['sep']
    inc_deg = normalize_inclination(row['inclination'])
    period_days = row['porb']
    r_mag_obs = row['r_mag_observed']

    current_age_Myr = 13700
    t = (current_age_Myr - (row['tphys'] + row['tbirth'])) / 1000.0
    age1 = t + row['aj_1'] / 1000.0
    age2 = t + row['aj_2'] / 1000.0

    sbratio = compute_sbratio(row['mass_1'], age1, rad1, row['mass_2'], age2, rad2)
    if np.isnan(sbratio):
        return np.nan

    r1_norm = rad1 / sep
    r2_norm = rad2 / sep

    phase = np.linspace(0, 1, 500)
    time = phase * period_days
    t_zero = 0.2 * period_days

    try:
        flux = ellc.lc(time,
                       radius_1=r1_norm, radius_2=r2_norm,
                       sbratio=sbratio,
                       incl=inc_deg,
                       t_zero=t_zero,
                       period=period_days,
                       a=sep,
                       shape_1='sphere', shape_2='sphere')
    except Exception:
        return np.nan

    norm = flux[0]
    if norm <= 0:
        return np.nan
    flux_norm = flux / norm
    delta_mag = -2.5 * np.log10(flux_norm)
    actual_mag = r_mag_obs + delta_mag
    amplitude = np.max(actual_mag) - np.min(actual_mag)
    return amplitude

# ================= 主程序 =================
def main():
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"未找到颜色筛选文件: {h5_file}")
    df = pd.read_hdf(h5_file, key='conv')
    print(f"读取颜色筛选源数: {len(df)}")

    if not os.path.exists(snr_file):
        raise FileNotFoundError(f"未找到 SNR 文件: {snr_file}")
    with open(snr_file, 'r') as f:
        first_line = f.readline().strip()
        n_cols = len(first_line.split())
    col_names = ['idx'] + [f'col{i}' for i in range(n_cols-2)] + ['snr']
    df_snr = pd.read_csv(snr_file, sep=r'\s+', names=col_names, header=None)
    df_snr = df_snr[['idx', 'snr']].copy()
    df_snr['idx'] = df_snr['idx'].astype(int)

    df = df.reset_index(drop=True)
    df['original_idx'] = df.index
    df_merged = pd.merge(df, df_snr, left_on='original_idx', right_on='idx', how='inner')
    print(f"成功匹配 SNR 的源数: {len(df_merged)}")
    if len(df_merged) == 0:
        print("错误: 无法匹配 SNR，请检查文件对应关系")
        return

    print("开始计算光变幅度...")
    amplitudes = []
    for i, (_, row) in enumerate(df_merged.iterrows()):
        amp = compute_amplitude(row)
        amplitudes.append(amp)
        if (i+1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(df_merged)}")
    df_merged['amplitude'] = amplitudes
    df_merged = df_merged.dropna(subset=['amplitude', 'snr'])
    print(f"有效数据点数: {len(df_merged)}")

    # 统计满足 SNR>4 且 amplitude>0.02 的源数量
    mask = (df_merged['snr'] > 4) & (df_merged['amplitude'] > 0.02)
    count_selected = mask.sum()
    print(f"信噪比>4且光变幅度>0.02的源数量: {count_selected} / {len(df_merged)}")

    # 可选：打印这些源的索引或保存到文件
    selected_indices = df_merged.index[mask].tolist()
    if count_selected > 0:
        print(f"满足条件的源索引（在合并后 DataFrame 中的位置）: {selected_indices}")

    # ========== 六边形密度图（hexbin） ==========
    x = df_merged['snr']
    y = df_merged['amplitude']
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(x, y, gridsize=40, cmap='viridis', mincnt=1, linewidths=0.5, edgecolors='none')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Number of sources per bin', fontsize=11)

    ax.axhline(y=0.02, color='red', linestyle='--', linewidth=1.5, label='Amplitude threshold (0.02 mag)')
    ax.axvline(x=7, color='blue', linestyle='--', linewidth=1.5, label='SNR = 7')
    ax.set_xlabel('LISA SNR (4.5 years)', fontsize=12)
    ax.set_ylabel('Optical amplitude (peak‑to‑peak mag)', fontsize=12)
    ax.set_title(f'α = {alpha}, colour‑selected DWDs (LSST) – hexbin density', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    out_png = os.path.join(base_dir, f'hexbin_alpha_{alpha}.png')
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"图已保存至 {out_png}")

if __name__ == "__main__":
    main()

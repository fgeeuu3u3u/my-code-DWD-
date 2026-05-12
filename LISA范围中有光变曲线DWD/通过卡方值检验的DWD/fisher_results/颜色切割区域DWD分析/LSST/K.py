
# python K.py

#!/usr/bin/env python3
"""
绘制颜色筛选 DWD 的 K1, K2, logTeff1, logTeff2 的散点矩阵图（pairplot）。
"""

#!/usr/bin/env python3
"""
2×1 图：上：logTeff1 vs logTeff2（分类散点）；下：K1 vs K2，颜色 = LISA SNR，形状 = 分类
"""

#!/usr/bin/env python3
"""
2×1 图：上：logTeff1 vs logTeff2（分类散点）；下：K1 vs K2，颜色 = LISA SNR（固定范围 0–7），形状 = 分类
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ================= 文件路径 =================
file_path = "/home/zhao/cosmic/LISA范围中有光变曲线DWD/通过卡方值检验的DWD/fisher_results/颜色切割区域DWD分析/LSST/dwd_1_interpolated_color_selected.h5"

# ================= 物理常数 =================
G = 6.67430e-11
Msun = 1.989e30
day_to_sec = 86400.0
km_per_m = 1e-3

def K_amp(M1, M2, P_days, sin_i):
    M1_kg = M1 * Msun
    M2_kg = M2 * Msun
    P_sec = P_days * day_to_sec
    K_mps = (2 * np.pi * G / P_sec)**(1/3) * (M2_kg * sin_i) / (M1_kg + M2_kg)**(2/3)
    return K_mps * km_per_m

# ================= 阈值 =================
K_THRESH = 15.0      # km/s
TEFF_THRESH = 10000  # K

# ================= 读取数据 =================
df = pd.read_hdf(file_path, key='conv')
print(f"总源数: {len(df)}")

# 读取 SNR 文件
snr_path = file_path.replace('dwd_1_interpolated_color_selected.h5', 'SigmasAE_LSST_1.dat')
try:
    with open(snr_path, 'r') as f:
        first_line = f.readline().strip()
        n_cols = len(first_line.split())
    col_names = ['idx'] + [f'col{i}' for i in range(n_cols-2)] + ['snr']
    df_snr = pd.read_csv(snr_path, sep=r'\s+', names=col_names, header=None)
    df_snr['idx'] = df_snr['idx'].astype(int)
    snr_by_idx = dict(zip(df_snr['idx'], df_snr['snr']))
except Exception:
    print("警告: 未找到 SNR 文件，使用随机模拟 SNR（仅用于演示）")
    snr_by_idx = None

# 计算 K1, K2, 温度
sin_i = np.sin(df['inclination'])
K1 = K_amp(df['mass_1'], df['mass_2'], df['porb'], sin_i)
K2 = K_amp(df['mass_2'], df['mass_1'], df['porb'], sin_i)

if 'logTeff_1' in df.columns:
    logTeff1 = df['logTeff_1'].values
    logTeff2 = df['logTeff_2'].values
    Teff1 = 10**logTeff1
    Teff2 = 10**logTeff2
elif 'teff_1' in df.columns:
    Teff1 = df['teff_1'].values
    Teff2 = df['teff_2'].values
    logTeff1 = np.log10(Teff1)
    logTeff2 = np.log10(Teff2)
else:
    raise KeyError("未找到温度列 (logTeff_1 / teff_1)")

mask = np.isfinite(K1) & np.isfinite(K2) & np.isfinite(logTeff1) & np.isfinite(logTeff2)
K1 = K1[mask]
K2 = K2[mask]
logTeff1 = logTeff1[mask]
logTeff2 = logTeff2[mask]
Teff1 = Teff1[mask]
Teff2 = Teff2[mask]
orig_idx = df.index[mask].values

# SNR 匹配
if snr_by_idx is not None:
    snr = np.array([snr_by_idx.get(idx, np.nan) for idx in orig_idx])
else:
    snr = np.random.uniform(2, 15, size=len(K1))
mask_snr = np.isfinite(snr)
K1 = K1[mask_snr]; K2 = K2[mask_snr]; logTeff1 = logTeff1[mask_snr]; logTeff2 = logTeff2[mask_snr]
Teff1 = Teff1[mask_snr]; Teff2 = Teff2[mask_snr]; snr = snr[mask_snr]
print(f"有效点数（含 SNR）: {len(K1)}")

# 分类 SB1/SB2/Unknown
star1_ok = (K1 > K_THRESH) & (Teff1 > TEFF_THRESH)
star2_ok = (K2 > K_THRESH) & (Teff2 > TEFF_THRESH)
is_sb2 = star1_ok & star2_ok
is_sb1 = (star1_ok ^ star2_ok)
is_unknown = ~(is_sb1 | is_sb2)

print("\n===== RV 可探测性统计 =====")
print(f"SB1 (单线): {np.sum(is_sb1)}")
print(f"SB2 (双线): {np.sum(is_sb2)}")
print(f"Unknown: {np.sum(is_unknown)}")

# ================= 绘图 =================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9))

# ---- 上子图：温度 vs 温度（分类散点）----
for cat, mask, color, marker in [('SB1', is_sb1, 'blue', 'o'),
                                  ('SB2', is_sb2, 'red', 's'),
                                  ('Unknown', is_unknown, 'gray', 'x')]:
    ax1.scatter(logTeff1[mask], logTeff2[mask], c=color, marker=marker,
                s=15, alpha=0.7, edgecolors='none', label=cat)
ax1.set_xlabel(r'$\log T_{\mathrm{eff},1}$', fontsize=12)
ax1.set_ylabel(r'$\log T_{\mathrm{eff},2}$', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

# ---- 下子图：K1 vs K2，颜色 = SNR（范围 0–7，超出用箭头），形状 = 分类 ----
# 设置颜色映射范围 0–7
vmin, vmax = 0, 7
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.plasma

# 绘制散点（注意：超出范围的点会被裁剪到边界颜色，但颜色条会显示箭头）
sc = None
for mask, marker in [(is_sb1, 'o'), (is_sb2, 's'), (is_unknown, 'x')]:
    if np.any(mask):
        s = ax2.scatter(K1[mask], K2[mask], c=snr[mask], cmap=cmap, norm=norm,
                        marker=marker, s=20, alpha=0.8, edgecolors='none')
        if sc is None:
            sc = s

# 颜色条，添加两端箭头（extend='both'）
cbar = fig.colorbar(sc, ax=ax2, extend='both')
cbar.set_label('LISA SNR (4.5 years)', fontsize=11)
# 设置刻度，强调范围边界
cbar.set_ticks([0, 2, 4, 6, 7])

ax2.set_xlabel(r'$K_1$ (km/s)', fontsize=12)
ax2.set_ylabel(r'$K_2$ (km/s)', fontsize=12)
# 形状图例
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='SB1'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=8, label='SB2'),
                   Line2D([0], [0], marker='x', color='w', markerfacecolor='k', markersize=8, label='Unknown')]
ax2.legend(handles=legend_elements, loc='upper left', title='Classification')
ax2.grid(alpha=0.3)

plt.tight_layout()
out_pdf = file_path.replace('.h5', '_2x1_SNR_color_fixed_range.pdf')
plt.savefig(out_pdf, dpi=150)
plt.show()
print(f"图已保存至: {out_pdf}")

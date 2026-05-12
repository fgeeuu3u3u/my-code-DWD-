#!/usr/bin/env python3
"""
python 统计有光变曲线+SB.py
统计各 alpha 下，满足有效阈值条件的源中 SB1 和 SB2 的数量。
有效阈值条件：信噪比（snr） >= 给定的 rho_eff。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/home/zhao/cosmic/LISA范围中有光变曲线DWD/通过卡方值检验的DWD"
LISA_DIR = "/home/zhao/cosmic/LISA范围中有光变曲线DWD/通过卡方值检验的DWD/fisher_results"
alphas = ['0.1', '0.3', '1', '3']
rho_eff_dict = {'0.1':4.7948, '0.3':4.7851, '1':5.4323, '3':5.0686}

def count_stages(alpha):
    lc_path = f"{BASE_DIR}/dwd_{alpha}_interpolated_lightcurve.h5"
    df = pd.read_hdf(lc_path, key='conv')
    total_original = len(df)
    
    # 星等筛选
    df_mag = df[df['r_mag_observed'] < 22].copy()
    after_mag = len(df_mag)
    
    # 温度筛选
    from mag_interpolation_LSST_CSST import CoolingTrackInterpolator
    TRACK_DIR = "/home/zhao/cosmic/alpha=0.1/DWD/AllTables_with_LSST"
    interp = CoolingTrackInterpolator(TRACK_DIR, survey='LSST', verbose=False)
    Teff1 = 10**df_mag['logTeff_1']
    Teff2 = 10**df_mag['logTeff_2']
    mask_hot = (Teff1 > 10000) | (Teff2 > 10000)
    df_hot = df_mag[mask_hot].copy()
    after_temp = len(df_hot)
    
    # K值筛选
    G = 6.67430e-11; Msun=1.989e30; km_per_m=1e-3; day_to_sec=86400.0
    def K_primary(M1, M2, P_days, sin_i):
        M1_kg = M1 * Msun; M2_kg = M2 * Msun; P_sec = P_days * day_to_sec
        K_mps = (2*np.pi*G/P_sec)**(1/3)*(M2_kg*sin_i)/(M1_kg+M2_kg)**(2/3)
        return K_mps * km_per_m
    df_hot['M1'] = df_hot[['mass_1','mass_2']].max(axis=1)
    df_hot['M2'] = df_hot[['mass_1','mass_2']].min(axis=1)
    sin_i = np.sin(df_hot['inclination'])
    K1 = K_primary(df_hot['M1'], df_hot['M2'], df_hot['porb'], sin_i)
    K2 = K_primary(df_hot['M2'], df_hot['M1'], df_hot['porb'], sin_i)
    K_max = np.maximum(K1, K2)
    mask_k = K_max > 15
    df_k = df_hot[mask_k].copy()
    after_k = len(df_k)
    
    # LISA信噪比筛选
    lisa_path = f"{LISA_DIR}/SigmasAE_{alpha}.dat"
    tmp = pd.read_csv(lisa_path, sep='\s+', header=None)
    n_cols = tmp.shape[1]
    df_lisa = pd.read_csv(lisa_path, sep='\s+', header=None, usecols=[0, n_cols-1], names=['idx', 'snr'])
    df_k['row_idx'] = df_k.index
    df_merged = pd.merge(df_k, df_lisa, left_on='row_idx', right_on='idx', how='inner')
    after_snr = len(df_merged[df_merged['snr'] >= rho_eff_dict[alpha]])
    
    # SB1/SB2 数量（使用之前手动统计的结果）
    if alpha == '0.1':
        sb1, sb2 = 0, 0
    elif alpha == '0.3':
        sb1, sb2 = 0, 0
    elif alpha == '1':
        sb1, sb2 = 5, 0
    elif alpha == '3':
        sb1, sb2 = 1, 0
    else:
        sb1, sb2 = 0, 0
    
    return [total_original, after_mag, after_temp, after_k, after_snr, sb1, sb2]

stages = [
    'Original\n(LC)',
    r'$m_r < 22$',
    r'$T_{\mathrm{eff}} > 10^4\,\mathrm{K}$',
    r'$K > 15\,\mathrm{km/s}$',
    r'$\mathrm{SNR} \geq \rho_{\mathrm{eff}}$',
    'SB1',
    'SB2'
]

data = {}
for alpha in alphas:
    data[alpha] = count_stages(alpha)

x = np.arange(len(stages))
width = 0.2
fig, ax = plt.subplots(figsize=(14,6))
for i, alpha in enumerate(alphas):
    ax.bar(x + i*width, data[alpha], width, label=f'alpha={alpha}')
ax.set_xticks(x + width*1.5, stages, rotation=45, ha='center')
ax.set_yscale('log')
ax.set_ylabel('Number of DWDs')
ax.legend()
ax.grid(axis='y', alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('selection_funnel_keep_order.png', dpi=300)
plt.show()




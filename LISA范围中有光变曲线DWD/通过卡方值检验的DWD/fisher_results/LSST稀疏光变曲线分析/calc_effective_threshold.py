#!/usr/bin/env python3
"""
python calc_effective_threshold.py
绘制观测时间 T 与计数 N(T) 和有效阈值的关系图（单 alpha=1，双轴）
无倾角筛选，观测时间 1-6 年，平滑曲线，黑色左轴，红色虚线右轴
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ================= 可调参数 =================
ALPHA = "1"
COLOR_LEFT = 'black'          # 左轴曲线颜色
COLOR_RIGHT = 'red'           # 右轴曲线颜色
base_name = f"dwd_{ALPHA}_interpolated_lightcurve"

# 文件路径
LISA_FILE = f"/home/zhao/cosmic/LISA范围中有光变曲线DWD/通过卡方值检验的DWD/fisher_results/SigmasAE_{ALPHA}.dat"
LC_FILE   = f"/home/zhao/cosmic/LISA范围中有光变曲线DWD/通过卡方值检验的DWD/fisher_results/lc/fisher_7param_sec_{base_name}.csv"
ASTRO_FILE = f"/home/zhao/cosmic/LISA范围中有光变曲线DWD/通过卡方值检验的DWD/fisher_results/location+distance/lsst_astrometry_errors_{base_name}.csv"

DEG2RAD = np.pi / 180.0
SEC_PER_DAY = 86400.0

k = 6                                                  # 参数: f, iota, theta, phi, fdot, DL
chi2_p = chi2.ppf(0.95, k)                            # 6自由度 95% 分位数 ≈12.59
chi2_1sigma = chi2.ppf(0.68, k)                       # 6自由度 68% 分位数 ≈7.00

# 全局参数空间体积（六维）
f_min_global = 1e-4
f_max_global = 0.01
iota_min, iota_max = 0, np.pi
theta_min, theta_max = 0, np.pi
phi_min, phi_max = 0, 2*np.pi
fdot_min_global = -1e-7    # Hz/s
fdot_max_global = 0.0
DL_min_global = 0.1
DL_max_global = 100

V_5d = (f_max_global - f_min_global) * (iota_max - iota_min) * \
       (theta_max - theta_min) * (phi_max - phi_min) * \
       (DL_max_global - DL_min_global)
V_total = V_5d * (fdot_max_global - fdot_min_global)   # 六维总体积

T_ref_years = 4.5
T_ref_sec = T_ref_years * 365.25 * 86400

def read_lisa_file(fname):
    param_cols = [f'p{i}' for i in range(9)]
    err_cols = [f'e{i}' for i in range(9)] + ['sky_err']
    cov_cols = [f'cov{i}' for i in range(36)]
    all_cols = ['idx'] + param_cols + err_cols + cov_cols + ['snr']
    df = pd.read_csv(fname, sep='\s+', names=all_cols, skipinitialspace=True)
    cov_values = df[cov_cols].values
    df['cov_mat'] = [cov.reshape(6,6) for cov in cov_values]
    df['snr'] = df['snr']
    return df

def read_lc_file(fname):
    df = pd.read_csv(fname)
    df['period_days'] = df['period_sec'] / SEC_PER_DAY
    df['err_period_days'] = df['err_period_sec'] / SEC_PER_DAY
    df['incl_deg'] = df['incl']
    df['err_incl_deg'] = df['err_incl']
    
    P_days = df['period_days']
    f = 2.0 / (P_days * SEC_PER_DAY)
    sigma_logP = np.sqrt(df['cov_logP_logP'])
    df['sigma_f'] = f * sigma_logP
    
    P_sec = P_days * SEC_PER_DAY
    dPdt = df['dPdt_true']
    df['fdot'] = -2.0 * dPdt / (P_sec**2)
    df['sigma_fdot'] = (2.0 / (P_sec**2)) * df['err_dPdt']
    
    cov_incl_deg_logP = df['cov_incl_logP']
    cov_iota_rad_logP = cov_incl_deg_logP * DEG2RAD
    df['cov_f_iota'] = -f * cov_iota_rad_logP
    
    cov_iota_rad_dPdt = df['cov_incl_dPdt'] * DEG2RAD
    df['cov_fdot_iota'] = (-2.0 / (P_sec**2)) * cov_iota_rad_dPdt
    
    cov_logP_dPdt = df['cov_logP_dPdt']
    df['cov_f_fdot'] = f * (2.0 / (P_sec**2)) * cov_logP_dPdt
    return df

def read_astro_file(fname):
    df = pd.read_csv(fname)
    df['theta_err2'] = df['err_co_latitude_rad']**2
    df['phi_err2'] = df['err_longitude_rad']**2
    df['DL_err2'] = df['err_distance_abs_kpc']**2
    df['distance_kpc'] = df['distance_kpc']
    return df

def compute_optical_covariance(row):
    Sigma_opt = np.zeros((6,6))
    Sigma_opt[0,0] = row['sigma_f']**2
    sigma_iota = row['err_incl_deg'] * DEG2RAD
    Sigma_opt[1,1] = sigma_iota**2
    Sigma_opt[2,2] = row['theta_err2']
    Sigma_opt[3,3] = row['phi_err2']
    Sigma_opt[4,4] = row['sigma_fdot']**2
    Sigma_opt[5,5] = row['DL_err2']
    
    Sigma_opt[0,1] = Sigma_opt[1,0] = row['cov_f_iota']
    Sigma_opt[4,1] = Sigma_opt[1,4] = row['cov_fdot_iota']
    Sigma_opt[0,4] = Sigma_opt[4,0] = row['cov_f_fdot']
    return Sigma_opt

def consistency_volume(cov_mat):
    det = np.linalg.det(cov_mat)
    if det <= 0:
        return 0.0
    return (chi2_p / chi2_1sigma)**(k/2) * np.sqrt((2*np.pi)**k * det)

def compute_fc_scaled(rho, df_total, T_sec, T_ref_sec):
    scale_snr = np.sqrt(T_sec / T_ref_sec)
    scale_cov = T_ref_sec / T_sec
    total_vol = 0.0
    count = 0
    for _, row in df_total.iterrows():
        snr_cur = row['snr'] * scale_snr
        if snr_cur < rho:
            continue
        count += 1
        Sigma_lisa = row['cov_mat'] * scale_cov
        Sigma_opt = compute_optical_covariance(row)
        Sigma_total = Sigma_lisa + Sigma_opt
        total_vol += consistency_volume(Sigma_total)
    fc = total_vol / V_total
    return fc, total_vol, count

def solve_rho_eff_and_stats(df_total, T_sec, T_ref_sec, rho_min=4.0, rho_max=6.9, tol=1e-4):
    def func(rho):
        fc, _, _ = compute_fc_scaled(rho, df_total, T_sec, T_ref_sec)
        Gamma = 10 ** (2 * (7 - rho))
        if Gamma <= 1:
            return 1e10
        return 7 + np.log(fc) / np.log(Gamma) - rho

    f_min = func(rho_min)
    f_max = func(rho_max)
    # print(f"  rho_min={rho_min}, f(rho_min)={f_min:.6e}")
    # print(f"  rho_max={rho_max}, f(rho_max)={f_max:.6e}")
    
    if f_min * f_max > 0:
        if f_min < 0 and f_max < 0:
            rho_min = max(3.0, rho_min - 1)
            # print(f"  函数值均为负，减小 rho_min 至 {rho_min}")
            return solve_rho_eff_and_stats(df_total, T_sec, T_ref_sec, rho_min, rho_max, tol)
        else:
            # print("  函数值同号且无法调整，返回 NaN")
            return np.nan, np.nan, np.nan, np.nan

    for i in range(50):
        rho_mid = (rho_min + rho_max) / 2
        f_mid = func(rho_mid)
        # print(f"  迭代 {i+1}: rho={rho_mid:.6f}, f(rho)={f_mid:.6e}")
        if abs(f_mid) < tol:
            fc, total_vol, count = compute_fc_scaled(rho_mid, df_total, T_sec, T_ref_sec)
            # print(f"  收敛: rho_eff={rho_mid:.6f}, fc={fc:.4e}, count={count}")
            return rho_mid, fc, total_vol, count
        if f_mid * f_min < 0:
            rho_max = rho_mid
            f_max = f_mid
        else:
            rho_min = rho_mid
            f_min = f_mid
    rho_mid = (rho_min + rho_max) / 2
    fc, total_vol, count = compute_fc_scaled(rho_mid, df_total, T_sec, T_ref_sec)
    # print(f"  达到最大迭代次数: rho_eff={rho_mid:.6f}, fc={fc:.4e}, count={count}")
    return rho_mid, fc, total_vol, count

def main():
    # 增加观测时间点数至 30，使曲线更平滑
    T_years = np.linspace(1.0, 6.0, 30)
    
    print(f"\n处理 alpha = {ALPHA}")
    df_lisa = read_lisa_file(LISA_FILE)
    df_lc = read_lc_file(LC_FILE)
    df_astro = read_astro_file(ASTRO_FILE)
    
    # 合并数据（不进行倾角筛选）
    df_merged = pd.merge(df_lc, df_astro, left_on='idx', right_on='source_idx')
    df_total = pd.merge(df_merged, df_lisa, left_on='idx', right_on='idx')
    print(f"  共有 {len(df_total)} 个匹配的源")
    
    # 统计信噪比 > 7 的源数量
    snr_gt7_count = (df_total['snr'] > 7).sum()
    print(f"  其中信噪比 > 7 的源数量: {snr_gt7_count} / {len(df_total)}")
    
    count_eff, count_7, rho_eff = [], [], []
    for T in T_years:
        T_sec = T * 365.25 * 86400
        rho, fc, total_vol, cnt_eff = solve_rho_eff_and_stats(df_total, T_sec, T_ref_sec)
        count_eff.append(cnt_eff)
        rho_eff.append(rho)
        scale_snr = np.sqrt(T_sec / T_ref_sec)
        cnt7 = sum(1 for _, row in df_total.iterrows() if row['snr'] * scale_snr >= 7.0)
        count_7.append(cnt7)
    
    # ========== 单图双轴绘制 ==========
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # 左轴：黑色实线 (ρ_ast) 和黑色虚线 (SNR≥7)
    ax1.plot(T_years, count_eff, color='black', linestyle='-', linewidth=2,
             label=r'$\alpha={}$ ($\rho_\ast$)'.format(ALPHA))
    ax1.plot(T_years, count_7, color='black', linestyle='--', linewidth=2,
             label=r'$\alpha={}$ (SNR$\geq$7)'.format(ALPHA))
    ax1.set_xlabel('Observation time T (years)', fontsize=12)
    ax1.set_ylabel('Number of sources', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # 右轴：红色虚线 (ρ_ast)
    ax2 = ax1.twinx()
    ax2.plot(T_years, rho_eff, color='red', linestyle='--', linewidth=2,
             label=r'$\rho_\ast$')
    ax2.set_ylabel(r'$\rho_\ast$', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(3.0, 7.0)
    ax2.set_yticks([3, 5, 7])
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 平台点标注（红色五角星）
    plateau_idx = len(count_7) - 1
    for i in range(len(count_7)-1):
        if np.all(np.abs(np.array(count_7[i:]) - count_7[i]) <= 0.5):
            plateau_idx = i
            break
    T_plateau = T_years[plateau_idx]
    y_plateau = count_7[plateau_idx]
    # 使用 LaTeX 五角星符号，大小调大
    ax1.scatter(T_plateau, y_plateau, marker=r'$\star$', s=300, color='red', zorder=5, edgecolors='none')
    ax1.text(T_plateau + 0.1, y_plateau + 0.2, f'T = {T_plateau:.1f} yr', fontsize=9, color='red', ha='left', va='bottom')
    
    plt.tight_layout()
    plt.savefig('combined_alpha_1.png', dpi=600)
    plt.close()
    print("\n图像已保存：combined_alpha_1.png")


if __name__ == "__main__":
    main()

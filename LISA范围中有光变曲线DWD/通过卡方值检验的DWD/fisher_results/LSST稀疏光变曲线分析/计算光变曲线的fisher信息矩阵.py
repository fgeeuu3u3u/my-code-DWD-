#!/usr/bin/env python3
"""
python 计算光变曲线的fisher信息矩阵.py

光变曲线 Fisher 信息矩阵计算（7参数，含周期变化率）
- 使用LSST稀疏光变 + 高密度后续观测
- 考虑引力波导致的周期衰减（dP/dt 根据Peters公式计算）
- 输出倾角、初始周期、周期变化率的3x3协方差矩阵及误差
"""



import os
import numpy as np
import pandas as pd
import ellc
from scipy.interpolate import interp1d
from scipy.linalg import inv
from mag_interpolation_LSST_CSST import CoolingTrackInterpolator

# ========== 1. 常量配置 ==========
MJD_REF = 57000.0
SEC_PER_DAY = 86400.0
N_PHASE = 1000
EPS = 1e-6

# 高密度后续观测参数
N_NIGHTS = 25
NIGHT_DURATION_HOURS = 8.0
CADENCE_SEC = 30.0
NOISE_MAG = 0.01
INTERVAL_DAYS = 120
INTERVAL_RAND = 5

# 自适应正则化基础相对强度
ALPHA = 1e-10

# ========== 2. 辅助函数 ==========
def dPdt_from_masses(m1, m2, P_days):
    """Peters (1964) 引力波周期变化率，返回 秒/秒"""
    G = 6.67430e-11
    c = 299792458.0
    Msun = 1.98892e30
    day2sec = 86400.0
    m1_kg = m1 * Msun
    m2_kg = m2 * Msun
    Mc = (m1_kg * m2_kg)**(3/5) / (m1_kg + m2_kg)**(1/5)
    P_sec = P_days * day2sec
    dPdt = - (96/5) * (2*np.pi)**(8/3) * (G * Mc)**(5/3) / c**5 * P_sec**(-5/3)
    return dPdt

def get_source_params(row):
    """从 HDF5 行提取真值（周期单位：天）"""
    rad1, rad2 = row['rad_1'], row['rad_2']
    sep = row['sep']
    incl_deg = row['inclination'] * 180.0 / np.pi
    if incl_deg > 90.0:
        incl_deg = 180.0 - incl_deg
    Pdays = row['porb']
    rmag = row['r_mag_observed']
    t = (13700 - (row['tphys'] + row['tbirth'])) / 1000.0
    age1 = t + row['aj_1']/1000.0
    age2 = t + row['aj_2']/1000.0

    def get_mag(m, a, r):
        res = interp.interpolate(m, a, r)
        return res['r_mag'] if res else np.nan
    mag1 = get_mag(row['mass_1'], age1, rad1)
    mag2 = get_mag(row['mass_2'], age2, rad2)
    sbr = 10.0 ** (-0.4 * (mag2 - mag1))
    r1n = rad1 / sep
    r2n = rad2 / sep
    return {
        'incl': incl_deg,
        'Pdays': Pdays,
        'logP': np.log(Pdays),
        'r1n': r1n, 'r2n': r2n,
        'logr1': np.log(r1n), 'logr2': np.log(r2n),
        'sbr': sbr,
        'logsbr': np.log(sbr),
        'rmag': rmag,
        'sep': sep,
        't0': MJD_REF + 0.2 * Pdays,
    }

def make_lightcurve_shape(r1, r2, sbr, incl, P, rmag, a):
    """生成一个完整周期的星等 vs 相位插值函数（形状固定）"""
    phases = np.linspace(0, 1, N_PHASE)
    times = phases * P
    flux = ellc.lc(times, radius_1=r1, radius_2=r2, sbratio=sbr, incl=incl,
                   t_zero=0.2*P, period=P, a=a, shape_1='sphere', shape_2='sphere',
                   verbose=0)
    flux /= np.max(flux)
    mag = rmag - 2.5 * np.log10(flux)
    return interp1d(phases, mag, kind='linear', bounds_error=False,
                    fill_value=(mag[-1], mag[0]))

# ========== 3. 生成高密度后续观测（正确考虑掩食漂移） ==========
def generate_high_cadence(src, dPdt_true):
    """
    生成 25 个夜晚的密集连续观测（每夜 8 小时，30 秒间隔）。
    正确模拟周期衰减导致的掩食中心时刻漂移。
    """
    P0 = src['Pdays']
    t0_ref = src['t0']
    # 夜晚中心时间（用于调度，与掩食中心无关）
    intervals = np.random.normal(INTERVAL_DAYS, INTERVAL_RAND, N_NIGHTS-1)
    night_centers = [t0_ref]
    for dt in intervals:
        night_centers.append(night_centers[-1] + dt)

    T_night = NIGHT_DURATION_HOURS / 24.0
    N_per_night = int(T_night * SEC_PER_DAY / CADENCE_SEC)
    shape = make_lightcurve_shape(src['r1n'], src['r2n'], src['sbr'], src['incl'],
                                  P0, src['rmag'], src['sep'])

    mjd_all = []
    mag_all = []
    sigma_all = []
    for t_night in night_centers:
        # 计算离 t_night 最近的掩食中心时刻（考虑周期衰减）
        k = int(round((t_night - t0_ref) / P0))
        # 二次近似掩食中心时刻
        t_center = t0_ref + P0 * k + 0.5 * dPdt_true * k * (k-1)
        # 以掩食中心为中心，前后各 T_night/2 采样
        t_start = t_center - T_night/2
        t_end   = t_center + T_night/2
        times = np.linspace(t_start, t_end, N_per_night)
        for t in times:
            dt = t - t0_ref
            P_cur = P0 + dPdt_true * dt
            if P_cur <= 0:
                continue
            # 相位以 t_center 为零点
            phase = ((t - t_center) / P_cur) % 1.0
            mag = shape(phase) + np.random.normal(0, NOISE_MAG)
            mjd_all.append(t)
            mag_all.append(mag)
            sigma_all.append(NOISE_MAG)
    return np.array(mjd_all), np.array(mag_all), np.array(sigma_all)

# ========== 4. Fisher 矩阵构建 ==========
def compute_fisher(src, obs_lsst_path, dPdt_true):
    """构建 7x7 数据 Fisher 矩阵，不加任何全局正则化"""
    # 读取 LSST 稀疏数据
    df = pd.read_csv(obs_lsst_path, skiprows=1)
    mjd_lsst = df['MJD'].values
    mag_lsst = df['mag_obs'].values
    sigma_lsst = df['sigma_mag'].values

    # 生成高密度数据
    mjd_high, mag_high, sigma_high = generate_high_cadence(src, dPdt_true)

    # 合并数据
    mjd = np.concatenate([mjd_lsst, mjd_high])
    mag_obs = np.concatenate([mag_lsst, mag_high])
    sigma = np.concatenate([sigma_lsst, sigma_high])

    # 真值参数向量 [incl, logP, dPdt, logr1, logr2, logsbr, t0]
    theta0 = np.array([
        src['incl'], src['logP'], dPdt_true,
        src['logr1'], src['logr2'], src['logsbr'], src['t0']
    ])

    def model(theta, mjd):
        incl, logP, dPdt, logr1, logr2, logsbr, t0 = theta
        P0 = np.exp(logP)
        if abs(dPdt) > 1e-6:
            return np.full_like(mjd, 1e6)
        r1, r2, sbr = np.exp(logr1), np.exp(logr2), np.exp(logsbr)
        shape = make_lightcurve_shape(r1, r2, sbr, incl, P0, src['rmag'], src['sep'])
        mags = np.zeros_like(mjd)
        for i, t in enumerate(mjd):
            dt = t - t0   # 相对参考掩食中心的时间差
            # 计算从参考掩食中心到当前时刻经过的整数圈数
            # 使用二次近似：k = (t - t0)/P0 - 0.5*dPdt*(t-t0)^2/P0^2
            # 更精确的方法：解方程，但二次近似足够
            # 为了数值稳定性，直接用简单方法：
            k = int(np.floor((t - t0) / P0))
            # 掩食中心时刻
            t_center = t0 + P0 * k + 0.5 * dPdt * k * (k-1)
            # 当前时刻的周期
            dt_cur = t - t0
            P_cur = P0 + dPdt * dt_cur
            if P_cur <= 0:
                return np.full_like(mjd, 1e6)
            phase = ((t - t_center) / P_cur) % 1.0
            mags[i] = shape(phase)
        return mags

    n_params = 7
    grad = np.zeros((len(mjd), n_params))

    # 梯度计算：dPdt 用绝对步长，其他用相对步长
    for i in range(n_params):
        val = theta0[i]
        if i == 2:   # dPdt
            step = 1e-15
        else:
            step = EPS * (1.0 + abs(val))
        theta_p = theta0.copy()
        theta_m = theta0.copy()
        theta_p[i] += step
        theta_m[i] -= step
        m_p = model(theta_p, mjd)
        m_m = model(theta_m, mjd)
        grad[:, i] = (m_p - m_m) / (2.0 * step)

    # 数据 Fisher 矩阵
    F = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            F[i, j] = np.sum((grad[:, i] * grad[:, j]) / (sigma**2))

    # 不添加任何全局正则化
    return F, theta0

# ========== 5. 提取核心参数协方差（分块法 + 自适应正则化） ==========
def get_core_covariance(F, alpha=ALPHA):
    """
    分块法边缘化干扰参数，仅对干扰参数子矩阵添加自适应正则化。
    核心参数（前3个）不添加任何正则化。
    """
    F11 = F[:3, :3]
    F12 = F[:3, 3:]
    F21 = F[3:, :3]
    F22 = F[3:, 3:]

    # 自适应正则化强度
    diag_mean = np.mean(np.diag(F22))
    reg = alpha * diag_mean
    reg_used = reg
    for attempt in range(5):
        try:
            F22_reg = F22 + reg_used * np.eye(4)
            F22_inv = inv(F22_reg)
            break
        except np.linalg.LinAlgError:
            reg_used *= 10
    else:
        raise np.linalg.LinAlgError("F22 正则化后仍奇异")

    F_eff = F11 - F12 @ F22_inv @ F21
    cov_core = inv(F_eff)   # 核心参数无正则化
    errs = np.sqrt(np.diag(cov_core))
    return cov_core, errs

# ========== 6. 主程序 ==========
def main():
    base = "/home/zhao/cosmic/LISA范围中有光变曲线DWD"
    passed = os.path.join(base, "通过卡方值检验的DWD")
    lightcurves = os.path.join(passed, "lightcurves")
    out_dir = os.path.join(passed, "fisher_results")
    os.makedirs(out_dir, exist_ok=True)

    global interp
    track_dir = "/home/zhao/cosmic/alpha=0.1/DWD/AllTables_with_LSST"
    interp = CoolingTrackInterpolator(track_dir, survey='LSST', verbose=False)

    h5_files = [f for f in os.listdir(passed) if f.endswith('.h5')]
    for h5 in h5_files:
        top = os.path.splitext(h5)[0]
        df = pd.read_hdf(os.path.join(passed, h5), key='conv')
        print(f"\n处理 {top}: {len(df)} 个源")
        results = []
        for idx, row in df.iterrows():
            try:
                src = get_source_params(row)
                dPdt_true = dPdt_from_masses(row['mass_1'], row['mass_2'], src['Pdays'])
                if dPdt_true > 0:
                    continue
            except Exception:
                continue
            if src['Pdays'] > 10.0:
                continue
            obs_path = os.path.join(lightcurves, top, f"source_{idx}", "obs.csv")
            if not os.path.exists(obs_path):
                continue
            print(f"  源 {idx}: i={src['incl']:.1f}°, P={src['Pdays']:.5f}d, dP/dt={dPdt_true:.2e} s/s")
            F, _ = compute_fisher(src, obs_path, dPdt_true)
            try:
                cov_core, errs = get_core_covariance(F)
            except np.linalg.LinAlgError as e:
                print(f"    警告: 源 {idx} Fisher 矩阵奇异 ({e})，跳过")
                continue
            if not np.all(np.isfinite(errs)):
                print(f"    警告: 源 {idx} 误差无效，跳过")
                continue
            period_sec = src['Pdays'] * SEC_PER_DAY
            err_period_sec = period_sec * errs[1]
            results.append({
                'idx': idx, 'incl': src['incl'], 'period_sec': period_sec,
                'dPdt_true': dPdt_true,
                'err_incl': errs[0], 'err_period_sec': err_period_sec, 'err_dPdt': errs[2],
                'cov_incl_incl': cov_core[0,0], 'cov_incl_logP': cov_core[0,1],
                'cov_incl_dPdt': cov_core[0,2], 'cov_logP_logP': cov_core[1,1],
                'cov_logP_dPdt': cov_core[1,2], 'cov_dPdt_dPdt': cov_core[2,2],
            })
            print(f"    误差: incl={errs[0]:.2f}°, P={err_period_sec:.3e}s, dPdt={errs[2]:.3e}")
        if results:
            df_out = pd.DataFrame(results)
            out_csv = os.path.join(out_dir, f"fisher_7param_sec_{top}.csv")
            df_out.to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()
    
    
    

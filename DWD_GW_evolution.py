'''
python DWD_GW_evolution.py
将DWD进行后续演化，只使用引力波衰减，发生过质量转移的后续的演化是不确定的，所以我们全部排除掉
'''
import os
import math
import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm

# =========================
# 常量定义
# =========================
G = 6.67430e-11
c = 2.99792458e8
MSUN = 1.98847e30
RSUN = 6.957e8
DAY = 86400.0
YEAR = 365.25 * DAY
MYR = 1e6 * YEAR
CHANDRASEKHAR_MASS = 1.44 * MSUN
CURRENT_AGE_MYR = 13700.0

# =========================
# 物理计算函数
# =========================
def calculate_wd_radius(mass):
    if mass <= 0: return 0.0
    m = mass * MSUN
    m_ratio = CHANDRASEKHAR_MASS / m
    term = m_ratio**(2/3) - (1/m_ratio)**(2/3)
    return max(1.4e-5, 0.0115 * math.sqrt(max(term, 0.0)))

def beta_factor(m1, m2):
    m1_kg = m1 * MSUN
    m2_kg = m2 * MSUN
    return (64/5) * G**3 / c**5 * m1_kg * m2_kg * (m1_kg + m2_kg)

def orbital_evolution_circular(a0, m1, m2, t_remain):
    if a0 <= 0 or t_remain <= 0: return a0, np.nan
    a0_si = a0 * RSUN
    beta_val = beta_factor(m1, m2)
    decay_term = 4 * beta_val * (t_remain * MYR)
    if decay_term >= a0_si**4: return 0.0, 0.0
    a_new_si = (a0_si**4 - decay_term)**0.25
    a_new = a_new_si / RSUN
    period_sec = 2 * math.pi * math.sqrt(a_new_si**3 / (G * ((m1+m2)*MSUN)))
    return a_new, period_sec / DAY

def roche_critical_sep(md, ma, rd):
    if ma <= 0 or rd <= 0: return float('inf')
    q = md / ma
    q23 = q**(2/3)
    q13 = q**(1/3)
    f = 0.49 * q23 / (0.6 * q23 + math.log(1 + q13))
    return rd / f

def decay_time(a0, a1, m1, m2):
    if a0 <= a1: return 0.0
    t = ((a0*RSUN)**4 - (a1*RSUN)**4) / (4 * beta_factor(m1, m2))
    return max(t / MYR, 0.0)

def will_undergo_rlo(a0, m1, m2, r1, r2, t_remain):
    a_critical_1 = roche_critical_sep(m1, m2, r1)
    a_critical_2 = roche_critical_sep(m2, m1, r2)
    t_rlo1 = decay_time(a0, a_critical_1, m1, m2) if a_critical_1 < a0 else float('inf')
    t_rlo2 = decay_time(a0, a_critical_2, m2, m1) if a_critical_2 < a0 else float('inf')
    return min(t_rlo1, t_rlo2) <= t_remain




# =========================
# 核心处理函数
# =========================
def process_single_row(row_dict):
    """处理单行数据，只保留分离系统并更新轨道参数和白矮星半径"""
    try:
        mass_1 = row_dict['mass_1']
        mass_2 = row_dict['mass_2']
        sep_original = row_dict['sep']
        porb_original = row_dict['porb']
        RRLO_1 = row_dict['RRLO_1']
        RRLO_2 = row_dict['RRLO_2']
        tbirth = row_dict['tbirth']
        tphys = row_dict['tphys']

        t_remain = CURRENT_AGE_MYR - (tbirth + tphys)

        # 检查是否已经发生质量转移
        if RRLO_1 >= 1.0 or RRLO_2 >= 1.0:
            return None  # 过滤掉质量转移系统

        # 计算白矮星半径并更新
        r1 = calculate_wd_radius(mass_1)
        r2 = calculate_wd_radius(mass_2)
        
        # 更新白矮星半径
        row_dict['rad_1'] = r1
        row_dict['rad_2'] = r2
        
        # 判断未来是否会发生RLOF
        if will_undergo_rlo(sep_original, mass_1, mass_2, r1, r2, t_remain):
            return None  

        # 分离系统：计算轨道演化后的新值
        sep_updated, porb_updated = orbital_evolution_circular(sep_original, mass_1, mass_2, t_remain)

        # 更新轨道参数
        row_dict['sep'] = sep_updated
        row_dict['porb'] = porb_updated

        return row_dict
        
    except Exception as e:
        print(f"Error processing system: {e}")
        return None


def process_batch(batch_data, column_names):
    """
    批量处理函数
    正确接收多个参数：batch_data, column_names
    """
    results = []
    
    for row_tuple in batch_data:
        # 将元组转换为字典
        row_dict = {}
        for i, col_name in enumerate(column_names):
            row_dict[col_name] = row_tuple[i]
        
        result = process_single_row(row_dict)
        if result is not None:  # 只保留非None结果（分离系统）
            results.append(result)
    
    return results



# =========================
# 主流程
# =========================
def main():
    input_root = "different_galactic_component_dwd_population"
    output_root = "different_galactic_component_dwd_population_detached"
    os.makedirs(output_root, exist_ok=True)

    components = [d for d in os.listdir(input_root) 
                 if os.path.isdir(os.path.join(input_root, d))]

    for comp in components:
        infile = os.path.join(input_root, comp, f"{comp}.h5")
        if not os.path.exists(infile):
            continue

        print(f"\nProcessing {comp} ...")
        outdir = os.path.join(output_root, comp)
        os.makedirs(outdir, exist_ok=True)

        out_h5 = os.path.join(outdir, f"{comp}_detached.h5")
        
        if os.path.exists(out_h5):
            os.remove(out_h5)

        # 优化参数配置
        chunk_size = 50000
        batch_size = 2000
        n_processes = min(mp.cpu_count(), 26)

        # 获取列名
        with pd.HDFStore(infile, 'r') as store:
            nrows = store.get_storer('conv').nrows
            sample_df = store.select('conv', start=0, stop=1)
            column_names = sample_df.columns.tolist()

        n_chunks = (nrows + chunk_size - 1) // chunk_size
        total_detached = 0
        total_processed = 0

        with Pool(processes=n_processes) as pool:
            for chunk_id in tqdm(range(n_chunks), desc=f"{comp} chunks"):
                start = chunk_id * chunk_size
                stop = min((chunk_id + 1) * chunk_size, nrows)

                # 读取当前数据块
                df_chunk = pd.read_hdf(infile, key='conv', start=start, stop=stop)
                if len(df_chunk) == 0:
                    continue

                total_processed += len(df_chunk)
                
                # 将数据转换为元组列表
                chunk_tuples = list(df_chunk.itertuples(index=False, name=None))
                
                # 将每个批次的数据构造为 (batch_data, column_names) 元组
                batches = []
                for i in range(0, len(chunk_tuples), batch_size):
                    batch = chunk_tuples[i:i + batch_size]
                    batches.append((batch, column_names))

                # 使用 starmap 处理批次
                batch_results = []
                for result in pool.starmap(process_batch, batches):
                    batch_results.extend(result)

                # 保存结果（只包含分离系统）
                if batch_results:
                    total_detached += len(batch_results)
                    df_result = pd.DataFrame(batch_results)
                    # 确保列顺序与原始数据一致
                    df_result = df_result.reindex(columns=column_names)
                    df_result.to_hdf(out_h5, key='conv', mode='a', format='table', append=True, index=False)

        print(f"  ✓ Completed: {comp}")
        print(f"    - Total systems processed: {total_processed}")
        print(f"    - Detached systems saved: {total_detached} ({total_detached/total_processed*100:.2f}%)")
        print(f"    - Mass-transfer systems filtered out: {total_processed - total_detached} ({(total_processed - total_detached)/total_processed*100:.2f}%)")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()


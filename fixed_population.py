#!/usr/bin/env python3
"""
python fixed_population.py
作者：zhao
银河系DWD数据处理脚本 - 双固定群体
核球、薄盘、厚盘使用群体A，晕使用群体B
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import gc
import psutil
import logging
from tqdm import tqdm
import warnings
import time
import multiprocessing as mp
from multiprocessing import Pool, Manager
import zlib

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

global_data = {}

class SequentialGalacticDWDProcessor:
    def __init__(self, output_base_dir="galactic_dwd_results",
                 batch_size=10000, memory_limit_gb=28, max_workers=None):
        self.output_base_dir = output_base_dir
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        self.max_workers = max_workers or max(1, mp.cpu_count() - 2)
        self.setup_directories()
        self.setup_galactic_parameters()

    def setup_directories(self):
        os.makedirs(self.output_base_dir, exist_ok=True)
        self.component_dirs = {}
        components = ['halo', 'thick_disc', 'bulge', 'thin_disc']
        for comp in components:
            comp_dir = os.path.join(self.output_base_dir, f'dwd_{comp}')
            os.makedirs(comp_dir, exist_ok=True)
            self.component_dirs[comp] = comp_dir
        logger.info(f"目录结构创建完成: {self.output_base_dir}")

    def setup_galactic_parameters(self):
        self.component_masses = {
            'halo': 1.0e9,
            'thick_disc': 2.6e9,
            'bulge': 2.0e10,
            'thin_disc': 5.2e10
        }
        self.component_populations = {
            'halo': 'population_B',
            'thick_disc': 'population_A',
            'bulge': 'population_A',
            'thin_disc': 'population_A'
        }
        self.spatial_params = {
            'bulge': {'r0': 0.5, 'rho0': 12.73},
            'thin_disc': {'hR': 2.5, 'hz': 0.352, 'rho0': 1.881},
            'thick_disc': {'hR': 2.5, 'hz': 1.158, 'rho0': 0.0286},
            'halo': {'a0': 2.7, 'rho0': 0.108}
        }
        self.current_age_myr = 13700

    def load_population_data(self, population_type):
        if population_type == 'population_A':
            conv_file = '../dat_kstar1_10_12_kstar2_10_12_SFstart_13700.0_SFduration_0_metallicity_0.02.h5'
        else:
            conv_file = '../dat_kstar1_10_12_kstar2_10_12_SFstart_13700.0_SFduration_0_metallicity_0.001.h5'

        logger.info(f"加载{population_type}数据: {conv_file}")
        with pd.HDFStore(conv_file, 'r') as store:
            conv = store['conv'][['bin_num','tphys','kstar_1','mass_1', 'kstar_2', 'mass_2', 'porb', 'sep', 'ecc', 'RRLO_1', 'RRLO_2', 'aj_1', 'aj_2','rad_1','rad_2']]
            mass_stars = store['mass_stars']

        total_mass = mass_stars.iloc[:, 0].max()
        logger.info(f"{population_type}: {len(conv)} 个系统, 总质量: {total_mass:.2e} M_sun")
        return conv, total_mass

    def check_memory_usage(self):
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        return memory_mb

    def init_worker(self, component_name, population_conv, population_total_mass, comp_mass):
        global global_data
        global_data = {
            'component_name': component_name,
            'population_conv': population_conv,
            'population_total_mass': population_total_mass,
            'comp_mass': comp_mass,
            'spatial_params': self.spatial_params,
            'current_age_myr': self.current_age_myr
        }

    def process_batch_worker(self, batch_info):
        try:
            batch_idx, batch_seed, output_queue = batch_info
            global global_data

            component_name = global_data['component_name']
            population_conv = global_data['population_conv']
            population_total_mass = global_data['population_total_mass']
            comp_mass = global_data['comp_mass']
            spatial_params = global_data['spatial_params']
            current_age_myr = global_data['current_age_myr']

            np.random.seed(batch_seed)

            N_component = int(len(population_conv) * (comp_mass / population_total_mass))
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, N_component)
            batch_size_actual = batch_end - batch_start

            if batch_size_actual <= 0:
                return None

            batch_sample = population_conv.sample(batch_size_actual, replace=True, random_state=batch_seed).copy()
            processed_batch = self._process_single_batch_worker(
                batch_sample, component_name, batch_size_actual, batch_seed, spatial_params, current_age_myr
            )

            if processed_batch is not None and len(processed_batch) > 0:
                # 直接放入写入队列：压缩后传输
                csv_bytes = processed_batch.to_csv(index=False).encode('utf-8')
                output_queue.put(zlib.compress(csv_bytes))

            del batch_sample
            gc.collect()
            return True

        except Exception as e:
            logger.error(f"批次 {batch_info[0]} 处理失败: {e}")
            return None

    def _process_single_batch_worker(self, batch_sample, component_name, batch_size, batch_seed, spatial_params, current_age_myr):
        np.random.seed(batch_seed)
        batch_sample = self._assign_birth_time_batch(batch_sample, component_name, batch_size)

        # 关键过滤：tphys + tbirth < 13700
        formation_time = batch_sample['tbirth'] + batch_sample['tphys']
        mask = formation_time < current_age_myr
        batch_sample_formed = batch_sample[mask].copy()

        if len(batch_sample_formed) == 0:
            return None

        x, y, z = self.sample_spatial_distribution_worker(component_name, len(batch_sample_formed), spatial_params)
        batch_sample_formed['x_pc'] = x
        batch_sample_formed['y_pc'] = y
        batch_sample_formed['z_pc'] = z

        return batch_sample_formed

    def _assign_birth_time_batch(self, batch_sample, component_name, batch_size):
        if component_name == 'halo':
            batch_sample['tbirth'] = 0
        elif component_name == 'thick_disc':
            batch_sample['tbirth'] = 3000
        else:
            t0, t_max = 4000, self.current_age_myr
            t_grid = np.linspace(t0, t_max, 10000)
            sfr = 11 * np.exp(-(t_grid - t0)/9000) + 0.12 * (t_grid - t0)
            cdf = np.cumsum(sfr)
            cdf /= cdf[-1]
            U = np.random.uniform(0, 1, batch_size)
            batch_sample['tbirth'] = np.interp(U, cdf, t_grid)

        return batch_sample

    def sample_spatial_distribution_worker(self, component_name, num_samples, spatial_params):
        params = spatial_params[component_name]
        if component_name == 'bulge':
            return self.sample_bulge_distribution(num_samples, params)
        elif component_name in ['thin_disc', 'thick_disc']:
            return self.sample_disc_distribution(num_samples, params, component_name)
        elif component_name == 'halo':
            return self.sample_halo_distribution(num_samples, params)

    def sample_bulge_distribution(self, num_samples, params):
        r0_pc = params['r0'] * 1000
        U = np.random.uniform(0, 1, num_samples)
        r_samples = -r0_pc * np.sqrt(np.log(1 / (1 - U + 1e-10)))
        theta = np.arccos(np.random.uniform(-1, 1, num_samples))
        phi = np.random.uniform(0, 2*np.pi, num_samples)
        x = r_samples * np.sin(theta) * np.cos(phi)
        y = r_samples * np.sin(theta) * np.sin(phi)
        z = r_samples * np.cos(theta)
        return x, y, z

    def sample_disc_distribution(self, num_samples, params, disc_type):
        hR_pc = params['hR'] * 1000
        hz_pc = params['hz'] * 1000
        U_R = np.random.uniform(0, 1, num_samples)
        R_samples = -hR_pc * np.log(1 - U_R)
        phi = np.random.uniform(0, 2*np.pi, num_samples)
        U_z = np.random.uniform(0, 1, num_samples)
        if disc_type == 'thin_disc':
            z_samples = hz_pc * np.arcsinh(np.tan(np.pi * (U_z - 0.5)))
        else:
            z_samples = -hz_pc * np.sign(U_z - 0.5) * np.log(1 - 2 * np.abs(U_z - 0.5))
        x = R_samples * np.cos(phi)
        y = R_samples * np.sin(phi)
        z = z_samples
        return x, y, z

    def sample_halo_distribution(self, num_samples, params):
        a0_pc = params['a0'] * 1000
        U = np.random.uniform(0, 1, num_samples)
        r_samples = a0_pc * np.tan(U * np.pi / 2)
        cos_theta = np.random.uniform(-1, 1, num_samples)
        phi = np.random.uniform(0, 2*np.pi, num_samples)
        x = r_samples * np.sqrt(1 - cos_theta**2) * np.cos(phi)
        y = r_samples * np.sqrt(1 - cos_theta**2) * np.sin(phi)
        z = r_samples * cos_theta
        return x, y, z

    def writer_process(self, component_name, output_queue, stop_event):
        comp_dir = self.component_dirs[component_name]
        h5_filename = os.path.join(comp_dir, f'dwd_{component_name}.h5')

        if os.path.exists(h5_filename):
            os.remove(h5_filename)

        # 创建表结构
        with pd.HDFStore(h5_filename, 'w') as store:
            store.put('conv', pd.DataFrame(), format='table', index=False)

        buffer = []
        while not stop_event.is_set() or not output_queue.empty():
            try:
                data = output_queue.get(timeout=1)
                csv_bytes = zlib.decompress(data)
                df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
                buffer.append(df)

                # 每累计 10 个 batch 写一次（降低IO）
                if len(buffer) >= 10:
                    with pd.HDFStore(h5_filename, 'a') as store:
                        store.append('conv', pd.concat(buffer, ignore_index=True), format='table', index=False)
                    buffer = []

            except Exception:
                pass

        # 写入剩余
        if buffer:
            with pd.HDFStore(h5_filename, 'a') as store:
                store.append('conv', pd.concat(buffer, ignore_index=True), format='table', index=False)

        logger.info(f"{component_name} 写入完成: {h5_filename}")

    def process_component_multiprocessing(self, component_name, population_conv, population_total_mass):
        comp_mass = self.component_masses[component_name]
        population_type = self.component_populations[component_name]

        logger.info(f"多进程处理 {component_name} - 使用{population_type}")
        logger.info(f"组分质量: {comp_mass:.2e} M_sun")
        logger.info(f"使用 {self.max_workers} 个工作进程")

        N_component = int(len(population_conv) * (comp_mass / population_total_mass))
        logger.info(f"需要抽样 {N_component} 个系统")
        if N_component == 0:
            return None

        num_batches = (N_component + self.batch_size - 1) // self.batch_size
        logger.info(f"总批次数: {num_batches}")

        manager = Manager()
        output_queue = manager.Queue(maxsize=50)  # 限制队列大小，防止堆积
        stop_event = manager.Event()

        writer = mp.Process(target=self.writer_process, args=(component_name, output_queue, stop_event))
        writer.start()

        base_seed = hash(component_name) % 10000 + 42
        batch_tasks = [(batch_idx, base_seed + batch_idx, output_queue) for batch_idx in range(num_batches)]

        try:
            with tqdm(total=num_batches, desc=f"多进程处理{component_name}") as pbar:
                with Pool(processes=self.max_workers,
                          initializer=self.init_worker,
                          initargs=(component_name, population_conv, population_total_mass, comp_mass),
                          maxtasksperchild=5) as pool:

                    for _ in pool.imap_unordered(self.process_batch_worker, batch_tasks, chunksize=1):
                        pbar.update(1)
                        if pbar.n % 10 == 0:
                            gc.collect()

        except Exception as e:
            logger.error(f"多进程处理失败: {e}")
            stop_event.set()
            writer.join()
            return None

        stop_event.set()
        writer.join()

        return os.path.join(self.component_dirs[component_name], f'dwd_{component_name}.h5')

    def count_rows_in_hdf(self, h5_path):
        count = 0
        with pd.HDFStore(h5_path, 'r') as store:
            for chunk in store.select('conv', chunksize=100000):
                count += len(chunk)
        return count

    def streaming_count_dwd_types(self, h5_path):
        counts = {'He+He':0,'He+CO':0,'CO+CO':0,'ONe+X':0}
        with pd.HDFStore(h5_path, 'r') as store:
            for chunk in store.select('conv', chunksize=100000):
                k1 = chunk['kstar_1'].astype(int)
                k2 = chunk['kstar_2'].astype(int)
                counts['He+He'] += int(((k1==10)&(k2==10)).sum())
                counts['He+CO'] += int((((k1==10)&(k2==11))|((k1==11)&(k2==10))).sum())
                counts['CO+CO'] += int(((k1==11)&(k2==11)).sum())
                counts['ONe+X'] += int(((k1==12)|(k2==12)).sum())
        return counts

    def generate_summary_report(self, results):
        report_file = os.path.join(self.output_base_dir, 'dwd_four_types_statistics.txt')
        core_dwd_types = ['He+He', 'He+CO', 'CO+CO', 'ONe+X']

        total_systems_all = 0
        component_stats = {}

        for comp_name, result in results.items():
            if result is None:
                component_stats[comp_name] = {'total':0,'core_type_counts':{t:0 for t in core_dwd_types}}
                continue

            h5_path = result
            total_systems = self.count_rows_in_hdf(h5_path)
            total_systems_all += total_systems
            core_type_counts = self.streaming_count_dwd_types(h5_path)

            component_stats[comp_name] = {
                'total': total_systems,
                'core_type_counts': core_type_counts
            }

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("银河系DWD系统四种核心类型统计报告\n")
            f.write("=" * 70 + "\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write(f"宇宙年龄: {self.current_age_myr} Myr\n\n")
            f.write(f"DWD系统总数: {total_systems_all:,}\n\n")

            for comp_name, stats in component_stats.items():
                f.write(f"{comp_name.upper()} 组分统计:\n")
                f.write("-"*60 + "\n")
                f.write(f"系统总数: {stats['total']:,}\n")
                if stats['total'] > 0:
                    comp_percentage = (stats['total']/total_systems_all)*100
                    f.write(f"占总体系比例: {comp_percentage:.2f}%\n\n")
                    f.write("四种核心DWD类型分布:\n")
                    for dwd_type in core_dwd_types:
                        count = stats['core_type_counts'][dwd_type]
                        percentage = (count/stats['total'])*100
                        f.write(f"  {dwd_type:6}: {count:>8} 系统 ({percentage:>6.2f}%)\n")
                else:
                    f.write("  无有效系统数据\n")
                f.write("\n")

            f.write("全局四种DWD类型汇总:\n")
            f.write("="*60 + "\n")
            global_type_totals = {t:0 for t in core_dwd_types}
            for stats in component_stats.values():
                for t in core_dwd_types:
                    global_type_totals[t] += stats['core_type_counts'][t]
            for t, c in sorted(global_type_totals.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{t:6}: {c:>8} 系统 ({(c/total_systems_all)*100:>6.2f}%)\n")

            f.write("\n总计: {:,} 个系统\n\n".format(total_systems_all))
            f.write("类型说明:\n")
            f.write("-"*60 + "\n")
            f.write("He WD: 氦核心白矮星 (kstar=10)\n")
            f.write("CO WD: 碳氧核心白矮星 (kstar=11)\n")
            f.write("ONe WD: 氧氖核心白矮星 (kstar=12)\n")
            f.write("ONe+X: 包含至少一个ONe WD的系统 (kstar_1=12 或 kstar_2=12)\n")
            f.write("注: He+CO 包含 (He+CO) 和 (CO+He) 两种组合\n")

        logger.info(f"四种DWD类型统计报告已保存: {report_file}")
        return report_file

    def run_multiprocessing_processing(self):
        logger.info("开始多进程处理银河系DWD数据")
        logger.info(f"使用 {self.max_workers} 个并行进程")

        popA_conv, popA_total_mass = self.load_population_data('population_A')
        popB_conv, popB_total_mass = self.load_population_data('population_B')

        results = {}
        components = ['bulge','thin_disc','thick_disc','halo']

        for comp_name in components:
            logger.info(f"\n{'='*60}")
            logger.info(f"开始处理 {comp_name} 组分")
            logger.info(f"{'='*60}")

            pop_type = self.component_populations[comp_name]
            population_conv = popA_conv if pop_type == 'population_A' else popB_conv
            population_total_mass = popA_total_mass if pop_type == 'population_A' else popB_total_mass

            h5_path = self.process_component_multiprocessing(comp_name, population_conv, population_total_mass)
            results[comp_name] = h5_path

            logger.info(f"完成处理 {comp_name}，清理内存...")
            time.sleep(2)
            gc.collect()
            logger.info(f"当前内存使用: {self.check_memory_usage():.2f} MB")

        self.generate_summary_report(results)
        return results


def main():
    output_dir = 'different_galactic_component_dwd_population'
    batch_size = 40000
    memory_limit_gb = 25
    max_workers = 28

    processor = SequentialGalacticDWDProcessor(
        output_base_dir=output_dir,
        batch_size=batch_size,
        memory_limit_gb=memory_limit_gb,
        max_workers=max_workers
    )

    results = processor.run_multiprocessing_processing()
    if results is not None:
        logger.info("多进程处理完成!")
        logger.info(f"所有结果保存在: {processor.output_base_dir}")

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
作者：zhao
python gbfisher_input_generate.py
根据不同银河系组分生成gbfisher输入文件，使用gbfisher中的Catalogue.c文件来进一步生成输入文件
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import os
import warnings
warnings.filterwarnings('ignore')

class ParallelGalaxyProcessor:
    def __init__(self, chunk_size=50000, n_processes=None):
        """
        初始化处理器
        
        参数:
        chunk_size: 分块大小，控制内存使用
        n_processes: 进程数，默认为CPU核心数-1
        """
        self.chunk_size = chunk_size
        # 设置CPU核数
        self.n_processes = n_processes or max(1, cpu_count() - 1)
        
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)
        
        # 定义处理顺序
        self.component_order = ['halo', 'thick_disc', 'thin_disc', 'bulge']
        print(f"将按顺序处理组分: {self.component_order}")
        print(f"设置使用 {self.n_processes} 个CPU核心")

    def _process_chunk(self, chunk_info):
        """
        处理单个数据块的函数，用于多进程
        """
        chunk_data, chunk_start, sun_distance, pure_name = chunk_info
        
        results = []
        n_sources = len(chunk_data)
        
        # 数据提取
        bin_num = chunk_data['bin_num'].values
        m1 = chunk_data['mass_1'].values
        m2 = chunk_data['mass_2'].values
        Porb = chunk_data['porb'].values
        ecc = chunk_data['ecc'].values
        x_pc = chunk_data['x_pc'].values
        y_pc = chunk_data['y_pc'].values
        z_pc = chunk_data['z_pc'].values
        
        # 坐标转换
        xgc = x_pc + sun_distance
        ygc = y_pc
        zgc = z_pc
        rec = np.sqrt(xgc**2 + ygc**2 + zgc**2)
        
        # 随机角度参数
        np.random.seed(42 + chunk_start)  # 确保可重复性
        iota = np.random.uniform(0, np.pi, n_sources)
        psi = np.random.uniform(0, 2*np.pi, n_sources)
        phi0 = np.random.uniform(0, 2*np.pi, n_sources)
        
        # 生成当前块的数据行
        for i in range(n_sources):
            global_index = chunk_start + i + 1
            line = f"{global_index}\t{m1[i]:.8f}\t{m2[i]:.8f}\t{Porb[i]:.8f}\t{ecc[i]:.8f}\t"
            line += f"{xgc[i]:.8f}\t{ygc[i]:.8f}\t{zgc[i]:.8f}\t{rec[i]:.8f}\t"
            line += f"{iota[i]:.8f}\t{psi[i]:.8f}\t{phi0[i]:.8f}\n"
            results.append(line)
        
        return results

    def process_single_component(self, base_project_dir, component_name):
        """
        使用多进程处理单个银河系组分
        """
        comp_path = Path(base_project_dir) / f"dwd_{component_name}"
        pure_name = component_name
        
        # 输入输出文件路径
        input_file = comp_path / f"dwd_{component_name}_detached.h5"
        output_file = comp_path / f"galaxy_{pure_name}.dat"
        
        print(f"\n开始处理组分: {pure_name}")
        
        if not input_file.exists():
            print(f"错误: 输入文件不存在: {input_file}")
            return False, 0
        
        try:
            # 获取总行数
            with pd.HDFStore(input_file, 'r') as store:
                total_rows = store.get_storer('conv').nrows
            
            print(f"数据总量: {total_rows:,} 行")
            
            # 坐标转换参数
            sun_distance = 8500
            
            # 打开输出文件
            with open(output_file, 'w') as outfile:
                # 添加银河系组分标识注释
                outfile.write(f"# Galactic Component: {pure_name}\n")
                outfile.write("# index\tmass1\tmass2\tporb\tecc\txGx\tyGx\tzGx\tdist\tinc\tOMEGA\tomega\n")
                
                # 创建进度条
                pbar = tqdm(total=total_rows, desc=f"处理 {pure_name}", 
                           unit='row', unit_scale=True, mininterval=1.0)
                
                # 分块读取和处理数据
                for chunk_start in range(0, total_rows, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, total_rows)
                    
                    with pd.HDFStore(input_file, 'r') as store:
                        # 读取当前块
                        conv_chunk = store.select('conv', 
                                                start=chunk_start, 
                                                stop=chunk_end,
                                                columns=['bin_num', 'mass_1', 'mass_2', 
                                                        'porb', 'ecc', 'x_pc', 'y_pc', 'z_pc'])
                    
                    n_sources = len(conv_chunk)
                    
                    # 如果数据量小，直接单进程处理
                    if n_sources < 10000 or self.n_processes == 1:
                        # 单进程处理
                        bin_num = conv_chunk['bin_num'].values
                        m1 = conv_chunk['mass_1'].values
                        m2 = conv_chunk['mass_2'].values
                        Porb = conv_chunk['porb'].values
                        ecc = conv_chunk['ecc'].values
                        x_pc = conv_chunk['x_pc'].values
                        y_pc = conv_chunk['y_pc'].values
                        z_pc = conv_chunk['z_pc'].values
                        
                        # 坐标转换
                        xgc = x_pc + sun_distance
                        ygc = y_pc
                        zgc = z_pc
                        rec = np.sqrt(xgc**2 + ygc**2 + zgc**2)
                        
                        # 随机角度参数
                        np.random.seed(42 + chunk_start)
                        iota = np.random.uniform(0, np.pi, n_sources)
                        psi = np.random.uniform(0, 2*np.pi, n_sources)
                        phi0 = np.random.uniform(0, 2*np.pi, n_sources)
                        
                        # 写入数据
                        for i in range(n_sources):
                            global_index = chunk_start + i + 1
                            outfile.write(f"{global_index}\t{m1[i]:.8f}\t{m2[i]:.8f}\t{Porb[i]:.8f}\t{ecc[i]:.8f}\t"
                                        f"{xgc[i]:.8f}\t{ygc[i]:.8f}\t{zgc[i]:.8f}\t{rec[i]:.8f}\t"
                                        f"{iota[i]:.8f}\t{psi[i]:.8f}\t{phi0[i]:.8f}\n")
                    else:
                        # 多进程处理 - 将当前块进一步分割
                        sub_chunk_size = max(1, n_sources // self.n_processes)
                        chunk_tasks = []
                        
                        for sub_start in range(0, n_sources, sub_chunk_size):
                            sub_end = min(sub_start + sub_chunk_size, n_sources)
                            sub_chunk = conv_chunk.iloc[sub_start:sub_end]
                            chunk_tasks.append((sub_chunk, chunk_start + sub_start, sun_distance, pure_name))
                        
                        # 使用进程池并行处理子块
                        with Pool(processes=min(self.n_processes, len(chunk_tasks))) as pool:
                            chunk_results = pool.map(self._process_chunk, chunk_tasks)
                            
                            # 合并结果并写入文件
                            for result in chunk_results:
                                for line in result:
                                    outfile.write(line)
                    
                    # 更新进度条
                    pbar.update(n_sources)
                    
                    # 清理内存
                    del conv_chunk
                
                # 关闭进度条
                pbar.close()
            
            # 获取实际处理的行数
            with open(output_file, 'r') as f:
                lines = f.readlines()
                actual_rows = len(lines) - 2  # 减去表头行和注释行
            
            print(f"✓ {pure_name} 处理完成: {actual_rows} 个源")
            return True, actual_rows
            
        except Exception as e:
            print(f"✗ 处理 {pure_name} 时出错: {e}")
            return False, 0

    def process_all_components_sequentially(self, base_project_dir):
        """
        按顺序处理所有银河系组分
        """
        base_path = Path(base_project_dir)
        
        if not base_path.exists():
            print(f"错误: 基础目录不存在: {base_path}")
            return {}
        
        print("开始顺序处理银河系各组分...")
        print("=" * 60)
        
        results = {}
        total_sources = 0
        
        # 按预定顺序逐个处理组分
        for component_name in self.component_order:
            comp_dir = base_path / f"dwd_{component_name}"
            
            if comp_dir.exists():
                success, num_sources = self.process_single_component(base_project_dir, component_name)
                results[component_name] = {'success': success, 'sources': num_sources}
                if success:
                    total_sources += num_sources
                
                print("-" * 40)
            else:
                print(f"警告: 组分目录不存在: {comp_dir}")
                results[component_name] = {'success': False, 'sources': 0}
        
        return results, total_sources

def main():
    """主函数"""
    base_directory = "different_galactic_component_dwd_population_detached"
    
    print("银河系组分数据处理脚本 - 多进程版本")
    print("=" * 60)
    

    processor = ParallelGalaxyProcessor(
        chunk_size=400000,  # 内存块大小
        n_processes=26    # CPU核心数
    )
    
    try:
        # 顺序处理所有银河系组分
        results, total_sources = processor.process_all_components_sequentially(base_directory)
        
        # 显示处理结果统计
        successful_components = sum(1 for result in results.values() if result['success'])
        total_components = len(results)
        
        print("\n" + "=" * 60)
        print("处理完成!")
        print(f"成功处理: {successful_components}/{total_components} 个组分")
        print(f"总源数量: {total_sources}")
        
        # 显示各组分统计
        print("\n各组分处理统计:")
        for comp, result in results.items():
            status = "✓" if result['success'] else "✗"
            print(f"  {status} {comp}: {result['sources']} 个源")
        
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()

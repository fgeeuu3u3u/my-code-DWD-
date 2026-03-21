#!/usr/bin/env python3
"""
作者：zhao
日期：2025.11.12
LSST白矮星绝对星等计算脚本
使用方法：python lsst_magnitudes_calculator.py
基于Rubin Simulator的phot_utils模块中的sed类来计算LSST每一个波段的绝对AB星等
输入：我们使用光谱插值脚本得到的La Plata冷却序列中的每一个白矮星的光谱文件
输出：每一个白矮星的基本数据+各个波段的绝对星等（AB星等）
注意：此时是没有加上dust map的！！！
"""

import os
import numpy as np
import glob
import logging
from rubin_sim.phot_utils import Bandpass, Sed
import re

# ================= 日志设置 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================= LSST 星等计算器 =================
class LSSTMagnitudeCalculator:
    """LSST绝对星等计算器 - 使用官方文档方法"""
    
    def __init__(self, throughputs_dir=None):
        self.logger = logging.getLogger(__name__)
        
        if throughputs_dir is None:
            throughputs_dir = os.getenv('LSST_THROUGHPUTS_BASELINE')
            if throughputs_dir is None:
                throughputs_dir = 'throughputs'
                if not os.path.exists(throughputs_dir):
                    raise ValueError(
                        "请设置LSST_THROUGHPUTS_BASELINE环境变量或提供throughputs_dir参数"
                    )
        
        self.throughputs_dir = throughputs_dir
        self.filterlist = ['u', 'g', 'r', 'i', 'z', 'y']
        self.bandpasses = self._load_bandpasses()
        self.logger.info("LSST滤波器加载完成")
    
    def _load_bandpasses(self):
        bandpasses = {}
        for filter_name in self.filterlist:
            bp = Bandpass()
            throughput_file = os.path.join(
                self.throughputs_dir, f'total_{filter_name}.dat'
            )
            
            if not os.path.exists(throughput_file):
                alternative_paths = [
                    os.path.join(self.throughputs_dir, 'baseline',
                                 f'total_{filter_name}.dat'),
                    os.path.join('/home/zhao/lsst/throughputs',
                                 f'total_{filter_name}.dat')
                ]
                for alt in alternative_paths:
                    if os.path.exists(alt):
                        throughput_file = alt
                        break
                else:
                    raise FileNotFoundError(f"找不到滤波器文件: {filter_name}")
            
            bp.read_throughput(throughput_file)
            bandpasses[filter_name] = bp
        
        return bandpasses
    
    def calculate_magnitudes_from_file(self, file_path):
        try:
            import tempfile
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.dat', delete=False
            ) as tf:
                for line in lines:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            wave_nm = float(parts[0]) / 10.0
                            flux_jy = float(parts[1]) * 1e23
                            tf.write(f"{wave_nm} {flux_jy}\n")
                        except ValueError:
                            continue
                tmpname = tf.name
            
            sed = Sed()
            sed.read_sed_fnu(tmpname)
            
            mags = {}
            for f, bp in self.bandpasses.items():
                try:
                    mags[f] = sed.calc_mag(bp)
                except Exception:
                    mags[f] = np.nan
            
            os.unlink(tmpname)
            return mags
        
        except Exception as e:
            self.logger.error(f"星等计算失败: {e}")
            return {f: np.nan for f in self.filterlist}


# ================= 核心处理函数 =================
def append_lsst_to_montreal_table(table_file, spectra_dir, output_dir, mag_calculator):
    """
    在蒙特利尔冷却表格文件后面追加LSST星等列
    按温度从高到低排序后一一对应
    """
    with open(table_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        logger.error(f"文件 {table_file} 行数不足")
        return 0, 0
    
    # 解析第一行获取质量
    first_line = lines[0].strip()
    parts_first = first_line.split()
    if len(parts_first) >= 2:
        try:
            mass = float(parts_first[1])
        except:
            mass = float(parts_first[0]) if parts_first[0].isdigit() else None
    else:
        mass = None
    
    if mass is None:
        logger.error(f"无法解析质量: {table_file}")
        return 0, 0
    
    # 加载该质量的光谱文件（按文件名排序，文件名是按Teff从高到低编号的）
    mass_str = f"{mass:.2f}Msun"
    mass_dir = os.path.join(spectra_dir, mass_str)
    
    if not os.path.exists(mass_dir):
        logger.error(f"光谱目录不存在: {mass_dir}")
        return 0, 0
    
    # 获取光谱文件列表（已按Teff从高到低排序，因为文件名 wd_spectrum_0001.dat 对应最高温）
    spectra_files = sorted(glob.glob(os.path.join(mass_dir, "wd_spectrum_*.dat")))
    
    if not spectra_files:
        logger.error(f"未找到质量 {mass} M☉ 的光谱文件")
        return 0, 0
    
    # 解析表格数据，按Teff从高到低排序
    data_rows = []
    for line_idx in range(2, len(lines)):
        line = lines[line_idx].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                teff = float(parts[0])
                data_rows.append({
                    'teff': teff,
                    'original_line': line,
                    'original_index': line_idx
                })
            except:
                continue
    
    if not data_rows:
        logger.warning(f"表格 {os.path.basename(table_file)} 没有有效数据行")
        return 0, 0
    
    # 按Teff从高到低排序
    data_rows_sorted = sorted(data_rows, key=lambda x: x['teff'], reverse=True)
    
    logger.info(f"处理 {os.path.basename(table_file)}: 质量={mass} M☉, 表格行数={len(data_rows)}, 光谱数={len(spectra_files)}")
    
    # 检查数量是否匹配
    if len(data_rows) != len(spectra_files):
        logger.warning(f"  数量不匹配: 表格 {len(data_rows)} 行, 光谱 {len(spectra_files)} 个")
    
    # 按顺序一一对应计算星等
    results = []
    for i, row in enumerate(data_rows_sorted):
        if i < len(spectra_files):
            spec_file = spectra_files[i]
            try:
                mags = mag_calculator.calculate_magnitudes_from_file(spec_file)
                results.append({
                    'original_index': row['original_index'],
                    'original_line': row['original_line'],
                    'mags': mags,
                    'success': True
                })
            except Exception as e:
                logger.error(f"  计算失败: Teff={row['teff']:.0f}K, {e}")
                results.append({
                    'original_index': row['original_index'],
                    'original_line': row['original_line'],
                    'mags': {b: np.nan for b in ['u','g','r','i','z','y']},
                    'success': False
                })
        else:
            # 光谱不足，填充nan
            logger.warning(f"  光谱不足: Teff={row['teff']:.0f}K 无对应光谱")
            results.append({
                'original_index': row['original_index'],
                'original_line': row['original_line'],
                'mags': {b: np.nan for b in ['u','g','r','i','z','y']},
                'success': False
            })
    
    # 按原始顺序重新排列
    results.sort(key=lambda x: x['original_index'])
    
    # 构建输出文件
    output_file = os.path.join(output_dir, os.path.basename(table_file))
    
    with open(output_file, 'w') as f:
        # 第一行
        f.write(first_line + '\n')
        
        # 第二行（添加LSST列标题）
        header_line = lines[1].rstrip()
        f.write(header_line + "      u      g      r      i      z      y\n")
        
        # 数据行
        success_count = 0
        for res in results:
            new_line = res['original_line']
            mags = res['mags']
            for band in ['u', 'g', 'r', 'i', 'z', 'y']:
                if np.isfinite(mags[band]):
                    new_line += f" {mags[band]:8.3f}"
                else:
                    new_line += "      nan"
            f.write(new_line + '\n')
            if res['success']:
                success_count += 1
        
        f.write('\n')
    
    logger.info(f"  完成: 成功={success_count}/{len(results)}")
    return success_count, len(results) - success_count


def process_all_tables(table_dir, spectra_dir, output_dir, throughputs_dir=None):
    """处理所有蒙特利尔冷却表格"""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("初始化LSST星等计算器...")
    mag_calculator = LSSTMagnitudeCalculator(throughputs_dir=throughputs_dir)
    
    table_files = sorted(glob.glob(os.path.join(table_dir, "Table_Mass_*")))
    if not table_files:
        raise RuntimeError(f"未找到表格文件: {table_dir}")
    
    logger.info(f"共找到 {len(table_files)} 个表格文件")
    
    total_success = 0
    total_failed = 0
    
    for tf in table_files:
        success, failed = append_lsst_to_montreal_table(
            tf, spectra_dir, output_dir, mag_calculator
        )
        total_success += success
        total_failed += failed
    
    logger.info(f"\n所有处理完成！总计: 成功={total_success}, 失败={total_failed}")


# ================= 主程序 =================
def main():
    TABLE_DIR = "/home/zhao/cosmic/alpha=0.1/DWD/AllTables"
    SPECTRA_DIR = "/home/zhao/cosmic/alpha=0.1/DWD/spectra_output"
    OUTPUT_DIR = "/home/zhao/cosmic/alpha=0.1/DWD/AllTables_with_LSST"
    THROUGHPUTS_DIR = None
    
    process_all_tables(TABLE_DIR, SPECTRA_DIR, OUTPUT_DIR, THROUGHPUTS_DIR)


if __name__ == "__main__":
    main()



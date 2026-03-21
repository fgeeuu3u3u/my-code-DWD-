#!/usr/bin/env python3
"""
白矮星光谱插值脚本
作者：zhao 
日期：2025-11-12
功能：使用Tremblay大气模型对La Plata不同核心成分的白矮星冷却表格中的白矮星进行光谱插值（温度范围1500-1400000K）
大气模型：
Main reference (3D corrections)
Tremblay, P.-E., Ludwig, H.-G., Steffen, M. & Freytag, B. (2013) A&A, 559, A104.
使用方法：python wd_spectrum_interpolation.py
输出：每一个白矮星的波长以及流量
波长单位：埃
温度范围：1500 to 140,000 K
流量单位：(units of erg cm^-2 s^-1 Hz^-1)
"""

import numpy as np
import os
import glob
import logging
from scipy.ndimage import gaussian_filter1d
import re

# ==================== 日志设置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 物理常数 ====================
PHYSICAL_CONSTANTS = {
    'L_sun': 3.828e33,           # erg/s
    'M_sun': 1.989e33,           # g
    'c': 2.99792458e10,          # cm/s
    'pc_to_cm': 3.08567758128e18, # cm/pc
    'Rsun_to_cm': 6.957e10,       # cm/R_sun
    'sigma': 5.670374419e-5,      # erg cm^-2 s^-1 K^-4
    'Mbol_sun': 4.74,             # mag
}

# ==================== Fortran 风格指数转换 ====================
def safe_float(s):
    """将 Fortran 风格的指数（0.12345-100）转换为 Python float"""
    s = s.strip()
    if re.match(r'^[\d.]+[-+]\d+$', s):
        s = s[:-len(re.findall(r'[-+]\d+$', s)[0])] + 'e' + re.findall(r'[-+]\d+$', s)[0]
    return float(s)


# ==================== 半径计算函数 ====================
def radius_from_mbol(teff, mbol):
    """
    从有效温度 (K) 和绝对热星等 (mag) 计算半径 (R_sun)
    """
    L_sun = PHYSICAL_CONSTANTS['L_sun']
    sigma = PHYSICAL_CONSTANTS['sigma']
    Mbol_sun = PHYSICAL_CONSTANTS['Mbol_sun']
    Rsun_cm = PHYSICAL_CONSTANTS['Rsun_to_cm']
    
    L_ratio = 10 ** (-0.4 * (mbol - Mbol_sun))
    L = L_sun * L_ratio
    R_cm = np.sqrt(L / (4 * np.pi * sigma * teff**4))
    R_sun = R_cm / Rsun_cm
    
    return R_sun


# ==================== Tremblay 模型插值器 ====================
class TremblayModelInterpolator:
    """Tremblay 白矮星大气模型插值器（纯氢DA模型）"""
    
    def __init__(self, model_dir, wave_points, wmin=1000, wmax=50000, dw=1.0, smooth=2.0):
        self.model_dir = model_dir
        self.wave_points = wave_points
        self.wmin = wmin
        self.wmax = wmax
        self.dw = dw
        self.smooth = smooth
        self.T_values = []
        self.logg_values = []
        self.model_data = []
        self._load_all_models()
        self._build_interpolator()
    
    def _load_all_models(self):
        logger.info(f"加载模型: {self.model_dir}")
        
        pattern = os.path.join(self.model_dir, '*_LyA_IR')
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"在 {self.model_dir} 中未找到模型文件")
        
        logger.info(f"找到 {len(files)} 个模型文件")
        
        for f in files:
            basename = os.path.basename(f)
            logg = float(basename.split('_')[0]) / 100.0
            
            with open(f, 'r') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
            
            if lines[0].isdigit():
                lines = lines[1:]
            
            # 读取波长
            wavelength_data = []
            idx = 0
            while len(wavelength_data) < self.wave_points:
                for val in lines[idx].split():
                    wavelength_data.append(safe_float(val))
                    if len(wavelength_data) >= self.wave_points:
                        break
                idx += 1
            wavelengths = np.array(wavelength_data)
            
            # 读取光谱
            i = idx
            while i < len(lines):
                line = lines[i]
                if 'Effective temperature' in line:
                    match = re.search(r'Effective temperature\s*=\s*([\d.]+).*gravity\s*=\s*([\d.E+-]+)', line)
                    if match:
                        teff = float(match.group(1))
                        flux_data = []
                        i += 1
                        while i < len(lines) and len(flux_data) < self.wave_points:
                            flux_data.extend([safe_float(x) for x in lines[i].split()])
                            i += 1
                        self.T_values.append(teff)
                        self.logg_values.append(logg)
                        self.model_data.append((wavelengths.copy(), np.array(flux_data)))
                    else:
                        i += 1
                else:
                    i += 1
        
        self.T_values = np.array(self.T_values)
        self.logg_values = np.array(self.logg_values)
        logger.info(f"加载完成: {len(self.T_values)} 个光谱")
    
    def _build_interpolator(self):
        self.wavelength = np.arange(self.wmin, self.wmax + self.dw, self.dw)
        self.T_grid = np.sort(np.unique(self.T_values))
        self.logg_grid = np.sort(np.unique(self.logg_values))
        n_T, n_logg, n_wl = len(self.T_grid), len(self.logg_grid), len(self.wavelength)
        self.spectra_grid = np.full((n_T, n_logg, n_wl), np.nan)
        
        for teff, logg, (wl_orig, flux_orig) in zip(self.T_values, self.logg_values, self.model_data):
            T_idx = np.where(self.T_grid == teff)[0][0]
            logg_idx = np.where(self.logg_grid == logg)[0][0]
            flux_interp = np.interp(self.wavelength, wl_orig, flux_orig)
            if self.smooth > 0:
                flux_interp = gaussian_filter1d(flux_interp, self.smooth / self.dw)
            self.spectra_grid[T_idx, logg_idx, :] = flux_interp
        
        logger.info(f"插值网格构建完成: Teff网格 {len(self.T_grid)} 个点, logg网格 {len(self.logg_grid)} 个点")
    
    def _bilinear_interpolation(self, teff, logg):
        """双线性插值获取四个角的权重"""
        T_idx1 = max(np.searchsorted(self.T_grid, teff, side='right') - 1, 0)
        T_idx2 = min(T_idx1 + 1, len(self.T_grid) - 1)
        logg_idx1 = max(np.searchsorted(self.logg_grid, logg, side='right') - 1, 0)
        logg_idx2 = min(logg_idx1 + 1, len(self.logg_grid) - 1)
        
        T1, T2 = self.T_grid[T_idx1], self.T_grid[T_idx2]
        logg1, logg2 = self.logg_grid[logg_idx1], self.logg_grid[logg_idx2]
        
        dx1 = (teff - T1) / (T2 - T1) if T2 != T1 else 0.0
        dy1 = (logg - logg1) / (logg2 - logg1) if logg2 != logg1 else 0.0
        dx2, dy2 = 1 - dx1, 1 - dy1
        
        return ([T1, logg1, dx2*dy2],
                [T2, logg1, dx1*dy2],
                [T1, logg2, dx2*dy1],
                [T2, logg2, dx1*dy1])
    
    def get_surface_flux(self, teff, logg):
        """获取表面通量 (erg/cm²/s/Hz)"""
        corners = self._bilinear_interpolation(teff, logg)
        flux = np.zeros_like(self.wavelength)
        total_weight = 0.0
        
        for T, logg_val, weight in corners:
            if weight > 1e-6:
                T_idx = np.where(self.T_grid == T)[0][0]
                logg_idx = np.where(self.logg_grid == logg_val)[0][0]
                flux += weight * self.spectra_grid[T_idx, logg_idx, :]
                total_weight += weight
        
        if total_weight > 0:
            flux /= total_weight
        
        return self.wavelength, flux
    
    def get_observed_flux(self, teff, logg, radius_Rsun, dist_pc=10.0):
        """获取观测通量 (erg/cm²/s/Hz) 在给定距离处"""
        wavelength, surface_flux = self.get_surface_flux(teff, logg)
        
        radius_cm = radius_Rsun * PHYSICAL_CONSTANTS['Rsun_to_cm']
        total_flux = surface_flux * 4 * np.pi * radius_cm**2
        
        dist_cm = dist_pc * PHYSICAL_CONSTANTS['pc_to_cm']
        observed_flux = total_flux / (4 * np.pi * dist_cm**2)
        
        return wavelength, observed_flux


# ==================== 蒙特利尔冷却表格读取器 ====================
class MontrealCoolingReader:
    """
    读取蒙特利尔冷却轨迹表格文件
    """
    
    def __init__(self, table_dir, mass_values=None):
        self.table_dir = table_dir
        self.mass_values = mass_values
        self.data = []
        self._load_all_tables()
    
    def _load_all_tables(self):
        if self.mass_values is None:
            pattern = os.path.join(self.table_dir, "Table_Mass_*")
            files = sorted(glob.glob(pattern))
            mass_list = []
            for f in files:
                basename = os.path.basename(f)
                mass_str = basename.replace("Table_Mass_", "")
                try:
                    mass = float(mass_str)
                    mass_list.append(mass)
                except:
                    continue
            self.mass_values = sorted(mass_list)
        
        logger.info(f"读取蒙特利尔冷却表格，质量: {self.mass_values}")
        
        for mass in self.mass_values:
            filename = os.path.join(self.table_dir, f"Table_Mass_{mass}")
            if not os.path.exists(filename):
                logger.warning(f"文件不存在: {filename}")
                continue
            
            logger.info(f"读取 {filename}")
            
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                logger.error(f"无法读取文件 {filename}: {e}")
                continue
            
            if len(lines) < 3:
                logger.error(f"文件 {filename} 行数不足")
                continue
            
            # 解析第一行，获取数据行数
            first_line = lines[0].strip()
            parts_first = first_line.split()
            if len(parts_first) > 0 and parts_first[0].isdigit():
                n_rows = int(parts_first[0])
            else:
                # 如果第一行不是数字，则从总行数估算
                n_rows = len(lines) - 2
            
            logger.info(f"  预期数据行数: {n_rows}")
            
            data_count = 0
            # 从第3行开始读取数据（索引2）
            for line_idx in range(2, len(lines)):
                line = lines[line_idx].strip()
                if not line:
                    continue
                
                parts = line.split()
                # 数据行至少有前3列（Teff, logg, Mbol）
                if len(parts) < 3:
                    continue
                
                try:
                    teff = float(parts[0])
                    logg = float(parts[1])
                    mbol = float(parts[2])
                    
                    radius = radius_from_mbol(teff, mbol)
                    
                    self.data.append({
                        'mass': mass,
                        'teff': teff,
                        'logg': logg,
                        'mbol': mbol,
                        'radius': radius,
                    })
                    data_count += 1
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"解析行 {line_idx+1} 失败: {e}")
                    continue
                
                # 如果已经读取到预期行数，提前结束
                if data_count >= n_rows:
                    break
            
            logger.info(f"  成功读取 {data_count} 个数据点")
        
        logger.info(f"加载完成: 总计 {len(self.data)} 个白矮星演化点")
    
    def get_all_parameters(self):
        return self.data
    
    def get_stars_by_mass(self):
        stars_by_mass = {}
        for s in self.data:
            mass_key = f"{s['mass']:.2f}Msun"
            stars_by_mass.setdefault(mass_key, []).append(s)
        return stars_by_mass


# ==================== 光谱保存 ====================
def save_spectrum(wavelength, flux, params, output_file):
    with open(output_file, 'w') as f:
        f.write(f"# Mass={params['mass']:.3f} M☉\n")
        f.write(f"# Teff={params['teff']:.0f} K\n")
        f.write(f"# logg={params['logg']:.3f}\n")
        f.write(f"# Radius={params['radius']:.5f} R☉\n")
        f.write(f"# Mbol={params['mbol']:.3f} mag\n")
        f.write(f"# Distance=10 pc\n")
        f.write("#\n")
        f.write("# Wavelength(A)  Flux(erg/cm2/s/Hz)\n")
        for wl, fl in zip(wavelength, flux):
            f.write(f"{wl:.1f}  {fl:.6e}\n")


# ==================== 汇总报告 ====================
def generate_summary_report(success_count, total_count, output_dir):
    report_file = os.path.join(output_dir, "processing_summary.txt")
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("白矮星光谱处理汇总报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"总白矮星演化点数: {total_count}\n")
        f.write(f"成功处理: {success_count}\n")
        if total_count > 0:
            f.write(f"成功率: {success_count/total_count*100:.1f}%\n")
        else:
            f.write("成功率: N/A (无数据)\n")
    logger.info(f"汇总报告生成: {report_file}")


# ==================== 主程序 ====================
def main():
    MODEL_DIR = "/home/zhao/cosmic/alpha=0.1/DWD/Tremblay_atm"
    TABLE_DIR = "/home/zhao/cosmic/alpha=0.1/DWD/AllTables"
    OUTPUT_DIR = "spectra_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("初始化 Tremblay 大气模型插值器...")
    model = TremblayModelInterpolator(MODEL_DIR, wave_points=2711)
    
    logger.info("读取蒙特利尔冷却表格...")
    cooling_reader = MontrealCoolingReader(TABLE_DIR)
    all_stars = cooling_reader.get_all_parameters()
    stars_by_mass = cooling_reader.get_stars_by_mass()
    
    success_count = 0
    total_count = len(all_stars)
    
    if total_count == 0:
        logger.error("没有读取到任何白矮星演化点！请检查文件路径和格式。")
        return
    
    logger.info(f"\n开始处理 {total_count} 个白矮星演化点...")
    
    for mass_str, stars in stars_by_mass.items():
        mass_dir = os.path.join(OUTPUT_DIR, mass_str)
        os.makedirs(mass_dir, exist_ok=True)
        
        stars_sorted = sorted(stars, key=lambda x: x['teff'], reverse=True)
        logger.info(f"处理质量 {mass_str}: {len(stars_sorted)} 个点")
        
        for i, params in enumerate(stars_sorted):
            try:
                wavelength, flux = model.get_observed_flux(
                    params['teff'], params['logg'], params['radius'], dist_pc=10.0
                )
                
                output_file = os.path.join(mass_dir, f"wd_spectrum_{i+1:04d}.dat")
                save_spectrum(wavelength, flux, params, output_file)
                success_count += 1
                
                if (i + 1) % 50 == 0:
                    logger.info(f"  {mass_str}: 已处理 {i+1}/{len(stars_sorted)} 个点")
                    
            except Exception as e:
                logger.error(f"质量 {mass_str} 光谱 {i+1} (Teff={params['teff']:.0f}K) 处理失败: {e}")
    
    generate_summary_report(success_count, total_count, OUTPUT_DIR)
    logger.info(f"处理完成: 成功 {success_count}/{total_count} 个白矮星")


if __name__ == "__main__":
    main()


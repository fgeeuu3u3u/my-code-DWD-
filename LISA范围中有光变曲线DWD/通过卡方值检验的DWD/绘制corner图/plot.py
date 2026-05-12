

# python plot.py



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from matplotlib.lines import Line2D

# 文件路径
files = {
    "alpha0.1_lightcurve": "/home/zhao/cosmic/LISA范围中有光变曲线DWD/通过卡方值检验的DWD/绘制corner图/dwd_1_interpolated_lightcurve.h5",
    "alpha1_highSNR": "/home/zhao/cosmic/LISA范围中有光变曲线DWD/通过卡方值检验的DWD/绘制corner图/dwd_1_interpolated_high_SNR.h5"
}

params = ['mass_1', 'mass_2', 'porb', 'logTeff_1', 'logTeff_2', 'd', 'inclination', 'r_mag_observed']
# 修改标签：周期单位改为分钟
labels = [r'$M_1\ (M_\odot)$', r'$M_2\ (M_\odot)$', r'$P_{\rm orb}\ ({\rm min})$',
          r'$\log T_{\rm eff,1}$', r'$\log T_{\rm eff,2}$', r'$d\ ({\rm kpc})$',
          r'$i\ (^\circ)$', r'$m_r$']

# 颜色：亮蓝色和亮橙色
colors = ['#1f77b4', '#ff7f0e']
legend_labels = ['DWD (LC)', 'DWD (high-SNR)']

data_arrays = []

for (name, fpath), color in zip(files.items(), colors):
    print(f"正在处理 {name} ...")
    if not os.path.exists(fpath):
        print(f"文件不存在: {fpath}")
        continue
    df = pd.read_hdf(fpath, key='conv')
    print(f"  原始数据行数: {len(df)}")
    # 将周期从天转换为分钟
    df['porb'] = df['porb'] * 24 * 60
    data = {}
    for col in params:
        if col not in df.columns:
            print(f"  警告: 缺少列 {col}，跳过此列")
            continue
        if col == 'inclination':
            data[col] = df[col].values * 180.0 / np.pi
        else:
            data[col] = df[col].values
    selected_cols = [c for c in params if c in df.columns]
    if not selected_cols:
        print("  没有可用的列，跳过")
        continue
    arr = np.column_stack([data[c] for c in selected_cols])
    arr = arr[~np.isnan(arr).any(axis=1)]
    print(f"  有效行数: {len(arr)}")
    data_arrays.append(arr)

if len(data_arrays) != 2:
    print("未能读取两个数据集，退出")
    exit()

# 绘制第一个数据集（蓝色）
fig = corner.corner(data_arrays[0], 
                    labels=labels[:data_arrays[0].shape[1]],
                    color=colors[0],
                    hist_kwargs={'density': True, 'histtype': 'step', 'edgecolor': colors[0], 'linewidth': 1.5},
                    show_titles=True,
                    title_kwargs={"fontsize": 10},
                    quantiles=[0.16, 0.5, 0.84],
                    label_kwargs={"fontsize": 12},
                    figsize=(12, 8))

# 叠加第二个数据集（橙色）
corner.corner(data_arrays[1], 
              fig=fig,
              color=colors[1],
              hist_kwargs={'density': True, 'histtype': 'step', 'edgecolor': colors[1], 'linewidth': 1.5},
              show_titles=False,
              label_kwargs={"fontsize": 12})

# 添加图例
legend_elements = [Line2D([0], [0], color=c, lw=2, label=l) for c, l in zip(colors, legend_labels)]
fig.axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10)

out_png = "combined_corner_density_step.png"   # 改为 PNG 格式
fig.savefig(out_png, dpi=300)
plt.show()
print(f"已保存 corner 图: {out_png}")


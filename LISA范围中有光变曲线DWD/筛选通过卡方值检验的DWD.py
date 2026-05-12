


#   python 筛选通过卡方值检验的DWD.py
#!/usr/bin/env python3
# 筛选通过卡方值检验的DWD.py（同时复制光变曲线和观测文件）

import os
import shutil
import pandas as pd
import glob

base_dir = "/home/zhao/cosmic/LISA范围中有光变曲线DWD"
output_root = os.path.join(base_dir, "通过卡方值检验的DWD")
lightcurves_dir = os.path.join(output_root, "lightcurves")
os.makedirs(output_root, exist_ok=True)
os.makedirs(lightcurves_dir, exist_ok=True)

# 找到所有结果文件
result_files = glob.glob(os.path.join(base_dir, "chi2_results_*.csv"))
print(f"找到 {len(result_files)} 个结果文件\n")

for res_file in result_files:
    # 提取顶层文件夹名
    top_folder = os.path.basename(res_file).replace("chi2_results_", "").replace(".csv", "")
    h5_file = os.path.join(base_dir, top_folder + ".h5")
    if not os.path.exists(h5_file):
        print(f"警告: 原始 HDF5 文件不存在 {h5_file}，跳过")
        continue

    # 读取结果文件，筛选通过检验的源（检测次数 >= 50）
    df_res = pd.read_csv(res_file)
    total = len(df_res)
    selected = df_res[df_res['detection_count'] >= 50]
    passed = len(selected)
    print(f"{top_folder}: 总源数={total}, 通过检验(≥50次检测)={passed}, 未通过={total-passed}")

    if passed == 0:
        print(f"  跳过 {top_folder}: 没有通过检验的源\n")
        continue

    # 获取通过检验的源序号
    keep_indices = [int(src.split('_')[1]) for src in selected['source'].values]

    # 1. 保存 HDF5 子集
    df_orig = pd.read_hdf(h5_file, key='conv')
    df_orig = df_orig.reset_index(drop=True)
    df_keep = df_orig.iloc[keep_indices].copy()
    out_h5 = os.path.join(output_root, top_folder + ".h5")
    df_keep.to_hdf(out_h5, key='conv', mode='w')
    print(f"  已保存通过检验的源参数到: {out_h5}")

    # 2. 复制光变曲线和观测文件
    out_lc_top = os.path.join(lightcurves_dir, top_folder)
    os.makedirs(out_lc_top, exist_ok=True)
    for idx in keep_indices:
        src_dir = os.path.join(base_dir, top_folder, f"source_{idx}")
        if not os.path.isdir(src_dir):
            print(f"  警告: 源目录不存在 {src_dir}，跳过")
            continue
        dst_dir = os.path.join(out_lc_top, f"source_{idx}")
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
    print(f"  已复制 {len(keep_indices)} 个源目录到 {out_lc_top}\n")

print("所有处理完成。")



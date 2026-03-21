#!/usr/bin/env python3
"""
由于网络问题数据需要手动放到dustmap3D的指定位置
简洁脚本：移动 dustmaps3d 数据文件到库指定位置
用法：python3 deploy_dustmaps_data.py
"""

import shutil
from pathlib import Path

def deploy_dustmaps_data():
    """移动数据文件到dustmaps3d库指定位置"""
    
    # 源文件（假设在当前目录）
    source_file = "data_v2.2.parquet"
    source_path = Path(source_file)
    
    # 获取库期望的目标路径
    try:
        from dustmaps3d.core import LOCAL_DATA_PATH
        target_path = LOCAL_DATA_PATH
        target_dir = target_path.parent
    except ImportError:
        print("错误: 无法导入 dustmaps3d，请确保库已正确安装")
        return False
    except Exception as e:
        print(f"错误: 获取库数据路径失败 - {e}")
        return False
    
    print(f"源文件: {source_path}")
    print(f"目标位置: {target_path}")
    print(f"目标目录: {target_dir}")
    
    # 检查源文件是否存在
    if not source_path.exists():
        print(f"错误: 源文件 '{source_file}' 不存在")
        print("请确保 data_v2.2.parquet 文件在当前目录中")
        return False
    
    try:
        # 创建目标目录（如果不存在）
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ 确保目录存在: {target_dir}")
        
        # 检查目标位置是否已存在文件
        if target_path.exists():
            print(f"警告: 目标位置已存在文件，将进行覆盖")
            # 可以选择备份原文件
            backup_path = target_path.with_suffix('.parquet.backup')
            shutil.move(target_path, backup_path)
            print(f"✓ 已备份原文件到: {backup_path}")
        
        # 直接移动文件（无需解压）
        print("移动文件中...")
        shutil.move(str(source_path), str(target_path))
        
        # 验证文件是否成功移动
        if target_path.exists() and target_path.stat().st_size > 0:
            file_size = target_path.stat().st_size
            print(f"✓ 成功! 文件已部署到: {target_path}")
            print(f"文件大小: {file_size} 字节 ({file_size/1024/1024:.2f} MB)")
            
            # 验证文件格式
            if verify_parquet_file(target_path):
                print("✓ Parquet文件格式验证通过")
            else:
                print("⚠ 文件格式验证失败，但文件已移动")
            
            return True
        else:
            print("✗ 错误: 文件移动失败")
            return False
            
    except Exception as e:
        print(f"✗ 处理过程中出错: {e}")
        return False

def verify_parquet_file(file_path):
    """验证Parquet文件格式是否正确"""
    try:
        import pandas as pd
        # 尝试读取Parquet文件验证格式
        df_test = pd.read_parquet(file_path, engine='pyarrow')
        print(f"✓ Parquet文件有效，包含 {len(df_test)} 行，{len(df_test.columns)} 列")
        return True
    except ImportError:
        print("⚠ 无法导入pandas，跳过Parquet格式验证")
        return True
    except Exception as e:
        print(f"⚠ Parquet文件验证失败: {e}")
        return False

def check_dustmaps3d_functionality():
    """检查dustmaps3d库是否正常工作"""
    try:
        from dustmaps3d import dustmaps3d
        # 简单的测试调用
        test_result = dustmaps3d([120.0], [30.0], [1.0])
        print("✓ dustmaps3d库功能正常")
        return True
    except Exception as e:
        print(f"⚠ dustmaps3d库测试调用失败: {e}")
        return False

if __name__ == "__main__":
    print("开始部署dustmaps3d数据文件...")
    print("=" * 50)
    
    if deploy_dustmaps_data():
        print("\n" + "=" * 50)
        print("部署完成!")
        print("=" * 50)
        
        # 检查库功能
        check_dustmaps3d_functionality()
        
        print("\n使用建议:")
        print("1. 现在可以正常运行您的dustmaps3d代码")
        print("2. 如果仍有问题，请检查文件权限和磁盘空间")
    else:
        print("\n部署失败，请检查错误信息")

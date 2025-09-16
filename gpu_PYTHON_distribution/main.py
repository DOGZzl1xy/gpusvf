# main.py
import time
import numpy as np
import argparse
import os
import glob

from utils import load_geotiff, save_raster, save_visualization
from gpusvf_calculator import svf_calculator_gpu, check_gpu_availability

def process_single_slice(dsm_path, tree_path, output_path):
    """
    处理单个文件对的核心逻辑
    
    参数:
    dsm_path: DSM GeoTIFF 文件路径，DEM+Building 数据
    tree_path: 植被 GeoTIFF 文件路径
    output_path: 输出 SVF GeoTIFF 文件路径
    
    """
    try:
        print(f"--- Processing slice: {os.path.basename(dsm_path)} ---")
        
        # 1. 数据加载
        dsm_img, gdal_dsm, scale = load_geotiff(dsm_path)
        tree_img, _, _ = load_geotiff(tree_path)
        tree_img[tree_img < 0] = 0

        print(f"  - Image shape: {dsm_img.shape}")

        # 2. 计算
        t_start = time.time()
        
        height, width = dsm_img.shape
        range_dist = int(np.sqrt(width**2 + height**2))

        res = svf_calculator_gpu(dsm_img, tree_img, np.float32(scale), range_dist)
        
        # 3. 后处理
        res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=0.0)
        res = np.clip(res, 0.0, 1.0)
        
        t_end = time.time()
        print(f'  - Computation time: {t_end - t_start:.2f} seconds')

        # 4. 保存结果
        save_raster(gdal_dsm, output_path, res)
        print(f"  - Saved result to {output_path}")
        
        # 保存可视化图像 (可选)
        # output_dir = os.path.dirname(output_path)
        # base_name = os.path.splitext(os.path.basename(output_path))[0]
        # viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        # save_visualization(res, viz_path)
        # print(f"  - Saved visualization to {viz_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process {os.path.basename(dsm_path)}. Reason: {e}")

def main():
    # --- 1. 参数设置 (在此处修改您的路径和文件名前后缀) ---
    input_dir = r"E:\Code\gpusvf\gpusvf\test_data\input"  # 包含输入切片的文件夹
    output_dir = r"E:\Code\gpusvf\gpusvf\test_data\output" # 保存结果的文件夹
    dsm_prefix = "BG_"     # DSM 文件的前缀
    tree_prefix = "Vege_"  # 植被文件的前缀
    file_extension = ".tif" # 文件扩展名

    # --- 2. GPU环境检查 ---
    if not check_gpu_availability():
        return

    # --- 3. 准备文件夹和查找文件 ---
    os.makedirs(output_dir, exist_ok=True)
    
    search_pattern = os.path.join(input_dir, f"{dsm_prefix}*{file_extension}")
    dsm_files = glob.glob(search_pattern)

    if not dsm_files:
        print(f"Warning: No DSM files found with prefix '{dsm_prefix}' in '{input_dir}'.")
        return

    print(f"Found {len(dsm_files)} DSM files to process.")

    # --- 4. 循环处理每个文件对 ---
    total_start_time = time.time()
    processed_count = 0

    for dsm_path in dsm_files:
        # 从DSM文件名推导其他文件名
        dsm_filename = os.path.basename(dsm_path)
        # 移除前缀和后缀，得到切片名 (例如 '1F')
        slice_name = dsm_filename.removeprefix(dsm_prefix).removesuffix(file_extension)
        
        tree_filename = f"{tree_prefix}{slice_name}{file_extension}"
        tree_path = os.path.join(input_dir, tree_filename)
        
        output_filename = f"{dsm_prefix}{slice_name}_svf.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        # 检查对应的植被文件是否存在
        if os.path.exists(tree_path):
            process_single_slice(dsm_path, tree_path, output_path)
            processed_count += 1
        else:
            print(f"Warning: Corresponding tree file '{tree_filename}' not found for {dsm_filename}. Skipping.")
    
    total_end_time = time.time()
    print("\n-----------------------------------------")
    print(f"Batch processing finished.")
    print(f"Successfully processed {processed_count} out of {len(dsm_files)} found files.")
    print(f"Total elapsed time: {total_end_time - total_start_time:.2f} seconds.")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()
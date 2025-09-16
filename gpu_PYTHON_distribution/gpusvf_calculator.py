# gpusvf_calculator.py
import cupy as cp
import numpy as np

def check_gpu_availability():
    """检查CuPy和CUDA是否可用，并打印GPU信息"""
    if not cp.cuda.is_available():
        print("CUDA is not available. Please check your CuPy installation.")
        return False
    
    print("CuPy version:", cp.__version__)
    print("CUDA available:", cp.cuda.is_available())
    device = cp.cuda.Device()
    print("GPU device count:", cp.cuda.runtime.getDeviceCount())
    print("Current device:", device.id)
    print(f"Total memory: {device.mem_info[1] / (1024**3):.2f} GB")
    print(f"Free memory: {device.mem_info[0] / (1024**3):.2f} GB")
    return True

def svf_calculator_gpu(dsm, tree, scale, range_dist): 
    """
    使用GPU加速的阴影投射算法计算天空开阔度因子
    
    参数：
    - dsm: numpy数组，DEM+Building数据
    - tree: numpy数组，Tree 表面高程
    - scale: float，数据的缩放因子（1/地图分辨率）
    - range_dist: int，计算范围距离（以像素为单位）

    返回：
    - numpy数组，计算得到的天空开阔度因子
    """
    # 从文件加载内核代码
    with open('svf_kernel.cu', 'r') as f:
        kernel_code = f.read()
    
    # 编译内核
    svf_kernel = cp.RawKernel(kernel_code, 'svf_shadowcasting_cupy')

    # 将输入数据移至GPU
    px_gpu = cp.asarray(dsm, dtype=cp.float32)
    tree_px_gpu = cp.asarray(tree, dtype=cp.float32)
    
    height, width = px_gpu.shape
    
    # 设置线程块和网格大小
    threads_per_block = (8, 8)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x)

    # 创建输出数组
    result_gpu = cp.empty_like(px_gpu)
    
    # 调用内核
    svf_kernel(
        blocks_per_grid,
        threads_per_block,
        (result_gpu, px_gpu, tree_px_gpu, cp.float32(scale), width, height, int(range_dist))
    )
    
    # 等待GPU计算完成并将结果移回CPU
    cp.cuda.Stream.null.synchronize()
    result_cpu = cp.asnumpy(result_gpu)
    
    return result_cpu
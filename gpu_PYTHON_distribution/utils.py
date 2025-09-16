# utils.py
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GDT_Float32
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='osgeo.gdal')

def load_geotiff(filepath):
    """
    加载并预处理GeoTIFF文件
    
    参数：
    - filepath: str，GeoTIFF文件的路径
    
    返回：
    - numpy数组，GeoTIFF数据
    - GDAL数据集对象，用于获取GeoTIFF的元数据
    - float，GeoTIFF数据的缩放因子
    """
    gdal_data = gdal.Open(filepath)
    if gdal_data is None:
        raise IOError(f"无法打开文件: {filepath}")

    band = gdal_data.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    
    data = band.ReadAsArray().astype(float)
    
    if nodata_value is not None:
        data = np.where(data == nodata_value, np.nan, data)
        
    # 用有效数据中的最小值填充NaN值
    if np.isnan(data).any():
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            min_valid = np.min(valid_data)
            data = np.nan_to_num(data, nan=min_valid)

    geotransform = gdal_data.GetGeoTransform()
    scale = 1 / geotransform[1] if geotransform[1] != 0 else 1.0

    return data, gdal_data, scale

def save_raster(gdal_template, filename, raster_data):
    """
    将numpy数组保存为GeoTIFF文件
    
    参数：
    - gdal_template: GDAL数据集对象，用于获取GeoTIFF的元数据
    - filename: str，保存的GeoTIFF文件名
    - raster_data: numpy数组，要保存的栅格数据
    
    """
    rows, cols = raster_data.shape
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, cols, rows, 1, GDT_Float32)
    out_band = out_ds.GetRasterBand(1)
    
    out_band.WriteArray(raster_data, 0, 0)
    out_band.FlushCache()
    out_band.SetNoDataValue(-9999)
    
    out_ds.SetGeoTransform(gdal_template.GetGeoTransform())
    out_ds.SetProjection(gdal_template.GetProjection())
    
    out_ds = None

def save_visualization(result_array, output_path):
    """
    保存结果的可视化PNG图像
    
    参数：
    - result_array: numpy数组，结果数据
    - output_path: str，保存PNG图像的路径
    """
    plt.imshow(result_array)
    plt.colorbar()
    plt.title("Sky View Factor")
    plt.savefig(output_path, dpi=300)
    plt.close()
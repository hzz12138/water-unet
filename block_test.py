import os
from osgeo import gdal
import psutil
import math
import numpy as np


def memory_usage():
    """计算可用内存"""
    mem_available = psutil.virtual_memory().available >> 20  # 可用内存
    mem_process = psutil.Process(os.getpid()).memory_info().rss >> 20  # 进程内存
    return mem_process, mem_available

def func_1(arr):
    """功能函数"""
    arr = arr.astype(np.float32)
    rst = (arr[6] - arr[2]) / ((arr[6] + arr[2]))
    rst = rst*1000
    rst = rst.astype(np.int16)
    return rst

def get_block(width, height, bands):
    # 计算分块数据
    # return: 分块个数，每块行数，剩余行数
    p, a = memory_usage()
    bl = (a - 2000) / (width * height * bands >> 20)
    if bl > 3:
        block_size = 1
    else:
        block_size = math.ceil(bl) + 4

    bl_height = int(height / block_size)
    mod_height = height % block_size

    return block_size, bl_height, mod_height


in_fn = r'E:\Zph\samples\images\suzhou.tif'
out_fn = r'E:\Zph\samples\images\suzhou_blocktest.tif'

ds = gdal.Open(in_fn)
width, height, bands = ds.RasterXSize, ds.RasterYSize, ds.RasterCount

# 分块
bl_size, bl_each, bl_mod = get_block(width, height, bands)
# 提取分块区域位置(起点,行数)
block_region = [(bs*bl_each, bl_each) for bs in range(bl_size)]
if bl_mod != 0:
    block_region.append([bl_size*bl_each, bl_mod])

# 输出结果保存
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(out_fn, width, height, 1, gdal.GDT_Int16)
out_ds.SetGeoTransform(ds.GetGeoTransform())
out_ds.SetProjection(ds.GetProjection())

# 分块计算并存入计算机
for h_pos, h_num in block_region:
    print(f'start height pos:{h_pos}, end height pos:{h_pos+h_num-1}')
    arr = ds.ReadAsArray(0, h_pos, width, h_num)
    rst = func_1(arr)
    out_ds.GetRasterBand(1).WriteArray(rst, 0, h_pos)

out_ds.FlushCache()
out_ds = None
print('ok')
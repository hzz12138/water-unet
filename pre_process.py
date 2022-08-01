from osgeo import gdal
import numpy as np
import datetime
import math
import sys
import torch
import cv2
from torchvision import transforms as T
import os
os.environ['PROJ_LIB'] = r'C:\Users\Lenovo\.conda\envs\zph\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\Lenovo\.conda\envs\zph\Library\share'
gdal.PushErrorHandler("CPLQuietErrorHandler")

class ImageProcess:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = gdal.Open(self.filepath, gdal.GA_ReadOnly)
        self.info = []
        self.img_data = None
        self.data_8bit = None

    def read_img_info(self):
        # 获取波段、宽、高
        img_bands = self.dataset.RasterCount
        img_width = self.dataset.RasterXSize
        img_height = self.dataset.RasterYSize
        # 获取仿射矩阵、投影
        img_geotrans = self.dataset.GetGeoTransform()
        img_proj = self.dataset.GetProjection()
        self.info = [img_bands, img_width, img_height, img_geotrans, img_proj]
        return self.info

    def read_img_data(self):
        self.img_data = self.dataset.ReadAsArray(0, 0, self.info[1], self.info[2])
        return self.img_data



    # 影像写入文件
    @staticmethod
    def write_img(filename: str, img_data: np.array, **kwargs):
        # 判断栅格数据的数据类型
        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(img_data.shape) >= 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        outdataset = driver.Create(filename, img_width, img_height, img_bands, datatype)
        # 写入仿射变换参数
        if 'img_geotrans' in kwargs:
            outdataset.SetGeoTransform(kwargs['img_geotrans'])
        # 写入投影
        if 'img_proj' in kwargs:
            outdataset.SetProjection(kwargs['img_proj'])
        # 写入文件
        if img_bands == 1:
            outdataset.GetRasterBand(1).WriteArray(img_data)  # 写入数组数据
        else:
            for i in range(img_bands):
                outdataset.GetRasterBand(i + 1).WriteArray(img_data[i])

        del outdataset


    def trans_img_16bit_to_8bit(self, low_per_raw=0.01,high_per_raw=0.99):
        array_data = self.img_data
        print(self.info[0])
        print(self.info[1])
        print(self.info[2])
        bands, cols, rows = self.info[0:3]  # array_data, (4, 36786, 37239) ,波段，行，列
        print("1读取数组形状", array_data.shape)
        # 这里控制要输出的是几位
        self.data_8bit = np.zeros((bands, rows, cols), dtype="uint8")
        array_data[array_data == 32767] = 0

        for i in range(bands):
            # 得到百分比对应的值，得到0代表的黑边的占比
            cnt_array = np.where(array_data[i, :, :], 0, 1)
            num0 = np.sum(cnt_array)
            #nodata_array = np.where(array_data[i, :, :] == 32767, 1, 0)
            #num1 = np.sum(nodata_array)
            kk = (num0) / (rows * cols)  # 得到0的比例
            #kk1 = (num1) / (rows * cols)  # 得到nodata的比例

            # 这里是去掉黑边0值，否则和arcgis的不一样，这样也有偏差，只算剩下的百分比
            low_per = low_per_raw + kk - low_per_raw * kk  # (A*x-A*KK)/(A-A*KK)=0.01, X = 0.99KK+0.01
            #low_per = low_per_raw + kk
            low_per = low_per * 100
            high_per = (1 - high_per_raw) * (1 - kk)  # A*x/(A-A*KK) = 0.04, X =  0.04-(0.04*kk)
            high_per = 100 - high_per * 100
            #high_per = kk1
            #high_per = high_per_raw *(100 - high_per * 100)
            print("baifen:", low_per, high_per)

            cutmin = np.percentile(array_data[i, :, :], low_per)
            cutmax = np.percentile(array_data[i, :, :], high_per)
            print("duandian:", cutmin, cutmax)

            data_band = array_data[i]
            # 进行截断
            data_band[data_band < cutmin] = cutmin
            data_band[data_band > cutmax] = cutmax
            # 进行缩放
            self.data_8bit[i, :, :] = np.around((data_band[:, :] - cutmin) * 255 / (cutmax - cutmin))

        print("最大最小值：", np.max(self.data_8bit), np.min(self.data_8bit))
        return self.data_8bit


if __name__ == '__main__':
    filepath = r'E:\Zph\0727琼中\mosaic.tif'
    #filepath = r'E:\Zph\Pytorch\data\image\bands_234_clip.tif'
    image = ImageProcess(filepath)
    # 获取影像信息
    info = image.read_img_info()
    print(type(info[4]), info[4])
    # 获取影像数据(np.array类型)
    data = image.read_img_data()
    # 将影像矩阵转为8bit
    data_8bit = image.trans_img_16bit_to_8bit()
    image.write_img(r'E:\Zph\0727琼中\mosaic_8bit.tif', data_8bit, img_geotrans=info[3], img_proj=info[4])

    # image_path = r'E:\Zph\Pytorch\Code\out\CB04A_mosaic_8bit.tif'
    # lable_path = r'E:\Zph\Pytorch\Code\out\water_label1.tif'

    print('finish')

import random
from osgeo import gdal
import numpy as np
import os
gdal.PushErrorHandler("CPLQuietErrorHandler")
import cv2

#  读取tif数据集
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


'''
随机裁剪函数
ImagePath 原始影像路径
LabelPath 标签影像路径
IamgeSavePath 原始影像裁剪后保存目录
LabelSavePath 标签影像裁剪后保存目录
CropSize 裁剪尺寸
CutNum 裁剪数量
'''


def RandomCrop(ImagePath, LabelPath, IamgeSavePath, LabelSavePath, CropSize, CutNum):
    im_width, im_height, im_bands, im_data, im_geotrans, im_proj = readTif(ImagePath)
    lab_width, lab_height, lab_bands, lab_data, lab_geotrans, lab_proj = readTif(LabelPath)

    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    fileNum = len(os.listdir(IamgeSavePath))
    new_name = fileNum + 1
    while (new_name < CutNum + fileNum + 1):
        #  生成剪切图像的左上角XY坐标
        UpperLeftX = random.randint(0, im_height - CropSize)
        UpperLeftY = random.randint(0, im_width - CropSize)
        if (len(im_data.shape) == 2):
            imgCrop = im_data[UpperLeftX: UpperLeftX + CropSize,
                      UpperLeftY: UpperLeftY + CropSize]
        else:
            imgCrop = im_data[:,
                      UpperLeftX: UpperLeftX + CropSize,
                      UpperLeftY: UpperLeftY + CropSize]
        if (len(lab_data.shape) == 2):
            labelCrop = lab_data[UpperLeftX: UpperLeftX + CropSize,
                        UpperLeftY: UpperLeftY + CropSize]
        else:
            labelCrop = lab_data[:,
                        UpperLeftX: UpperLeftX + CropSize,
                        UpperLeftY: UpperLeftY + CropSize]
        if (10000 * 255) < np.sum(labelCrop) < (220000 * 255):
            writeTiff(imgCrop, im_geotrans, im_proj, IamgeSavePath + "/%d.tif" % new_name)
            writeTiff(labelCrop, im_geotrans, im_proj, LabelSavePath + "/%d.tif" % new_name)
            new_name = new_name + 1



# changshu_CB04A_8bit songjiang_GF2_8bit suzhou_8bit taicang_8bit wuxi_8bit yangcheng_GF1_8bit yangcheng_GF2_8bit
# changshu_CB04A_label_8bit songjiang_GF2_label_8bit suzhou_label_8bit taicang_label_8bit wuxi_label_8bit yancheng_GF1_label_8bit yancheng_GF2_label_8bit
image_path = r"E:\Zph\samples\16to8\yangcheng_GF2_8bit.tif"
label_path = r"E:\Zph\samples\labels_8bit\yancheng_GF2_label_8bit.tif"
image_dir = r"E:\Zph\samples\1validation\images"
label_dir = r"E:\Zph\samples\1validation\labels"

#  裁剪得到300张512*512大小的训练集
RandomCrop(image_path, label_path, image_dir, label_dir, 512, 50)

# #  进行几何变换数据增强
# imageList = os.listdir(image_dir)
# labelList = os.listdir(label_dir)
# tran_num = len(imageList) + 1
# for i in range(len(imageList)):
#     #  图像
#     img_file = image_dir + "\\" + imageList[i]
#     im_width, im_height, im_bands, im_data, im_geotrans, im_proj = readTif(img_file)
#     #  标签
#     label_file = label_dir + "\\" + labelList[i]
#     lab_width, lab_height, lab_bands, lab_data, lab_geotrans, lab_proj = readTif(label_file)
#     # label = cv2.imread(label_file,0)
#
#     #  图像水平翻转
#     im_data_hor = np.flip(im_data, axis=2)
#     hor_path = image_dir + "\\" + str(tran_num) + imageList[i][-4:]
#     writeTiff(im_data_hor, im_geotrans, im_proj, hor_path)
#     #  标签水平翻转
#     Hor = np.flip(lab_data, axis=1)
#     hor_path = label_dir + "\\" + str(tran_num) + labelList[i][-4:]
#     writeTiff(Hor, lab_geotrans, lab_proj, hor_path)
#     tran_num += 1
#
#     #  图像垂直翻转
#     im_data_vec = np.flip(im_data, axis=1)
#     vec_path = image_dir + "\\" + str(tran_num) + imageList[i][-4:]
#     writeTiff(im_data_vec, im_geotrans, im_proj, vec_path)
#     #  标签垂直翻转
#     Vec = np.flip(lab_data, axis=0)
#     vec_path = label_dir + "\\" + str(tran_num) + labelList[i][-4:]
#     writeTiff(Vec, lab_geotrans, lab_proj, vec_path)
#     tran_num += 1
#
#     #  图像对角镜像
#     im_data_dia = np.flip(im_data_vec, axis=2)
#     dia_path = image_dir + "\\" + str(tran_num) + imageList[i][-4:]
#     writeTiff(im_data_dia, im_geotrans, im_proj, dia_path)
#     #  标签对角镜像
#     Dia = np.flip(Vec, axis=1)
#     dia_path = label_dir + "\\" + str(tran_num) + labelList[i][-4:]
#     writeTiff(Dia, lab_geotrans, lab_proj, dia_path)
#     tran_num += 1
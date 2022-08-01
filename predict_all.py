from osgeo import gdal
import numpy as np
import datetime
import math
import sys
import segmentation_models_pytorch as smp
import torch
import cv2
from torchvision import transforms as T
from UNet import Unet
gdal.PushErrorHandler("CPLQuietErrorHandler")
from UNetPP import NestedUNet
import argparse
from attention_unet import AttU_Net

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=21)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='resnet34_unet',
                       help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument('--dataset', default='driveEye',  # dsb2018_256
                       help='dataset name:liver/esophagus/dsb2018Cell/corneal/driveEye/isbiCell/kaggleLung')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    args = parse.parse_args()
    return args

# 读取tif数据集
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

# 保存tif文件函数
def writeTiff(fileName, data, im_geotrans=(0, 0, 0, 0, 0, 0), im_proj=""):
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(data.shape) == 3:
        im_bands, im_height, im_width = data.shape
    elif len(data.shape) == 2:
        data = np.array([data])
        im_bands, im_height, im_width = data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fileName, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(data[i])
    del dataset


#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (512 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (512 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (512 - SideLength * 2): i * (512 - SideLength * 2) + 512,
                      j * (512 - SideLength * 2): j * (512 - SideLength * 2) + 512]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (512 - SideLength * 2): i * (512 - SideLength * 2) + 512,
                  (img.shape[1] - 512): img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 512): img.shape[0],
                  j * (512 - SideLength * 2): j * (512 - SideLength * 2) + 512]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 512): img.shape[0],
              (img.shape[1] - 512): img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (512 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (512 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


#  获得结果矩阵
def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0
    for i, img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if (i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 512 - RepetitiveLength, 0: 512 - RepetitiveLength] = img[0: 512 - RepetitiveLength,
                                                                               0: 512 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                #  原来错误的
                # result[shape[0] - ColumnOver : shape[0], 0 : 512 - RepetitiveLength] = img[0 : ColumnOver, 0 : 512 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: 512 - RepetitiveLength] = img[
                                                                                                        512 - ColumnOver - RepetitiveLength: 512,
                                                                                                        0: 512 - RepetitiveLength]
            else:
                result[j * (512 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            512 - 2 * RepetitiveLength) + RepetitiveLength,
                0:512 - RepetitiveLength] = img[RepetitiveLength: 512 - RepetitiveLength, 0: 512 - RepetitiveLength]
                #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 512 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0: 512 - RepetitiveLength,
                                                                                  512 - RowOver: 512]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[512 - ColumnOver: 512,
                                                                                        512 - RowOver: 512]
            else:
                result[j * (512 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            512 - 2 * RepetitiveLength) + RepetitiveLength,
                shape[1] - RowOver: shape[1]] = img[RepetitiveLength: 512 - RepetitiveLength, 512 - RowOver: 512]
                #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 512 - RepetitiveLength,
                (i - j * len(TifArray[0])) * (512 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (512 - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[0: 512 - RepetitiveLength, RepetitiveLength: 512 - RepetitiveLength]
                #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0],
                (i - j * len(TifArray[0])) * (512 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (512 - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[512 - ColumnOver: 512, RepetitiveLength: 512 - RepetitiveLength]
            else:
                result[j * (512 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            512 - 2 * RepetitiveLength) + RepetitiveLength,
                (i - j * len(TifArray[0])) * (512 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (512 - 2 * RepetitiveLength) + RepetitiveLength,
                ] = img[RepetitiveLength: 512 - RepetitiveLength, RepetitiveLength: 512 - RepetitiveLength]
    return result



TifPath = r"C:\Users\Lenovo\Desktop\water_test\S04_8bit.tif"
ResultPath = r"C:\Users\Lenovo\Desktop\water_test\predict_attu_net.tif"

im_width, im_height, im_bands, big_image, im_geotrans, im_proj = readTif(TifPath)
big_image = big_image.swapaxes(1, 0).swapaxes(1, 2)

area_perc = 0.5
RepetitiveLength = int((1 - math.sqrt(area_perc)) * 512 / 2)
TifArray, RowOver, ColumnOver = TifCroppingArray(big_image, RepetitiveLength)

# unet model
# model = Unet(4,1)

# unetpp model
# args = getArgs()
# args.deepsupervision = False
# model = NestedUNet(args,4,1).cuda()

# attu_unet model
model = AttU_Net(4,1).cuda()

# 将模型加载到指定设备DEVICE上
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(DEVICE)
model.load_state_dict(torch.load('model/best_model_attu_net.pth', map_location=DEVICE))
model.eval()

predicts = []
for i in range(len(TifArray)):
    for j in range(len(TifArray[0])):
        image = TifArray[i][j].swapaxes(0,2).swapaxes(1,2)
        img = np.expand_dims(image, 0)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=DEVICE, dtype=torch.float32)
        pred = model(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        predicts.append(pred)
    print(f"""{i}/{len(TifArray)}""")

# 保存结果predictspredicts
result_shape = (big_image.shape[0], big_image.shape[1])
result_data = Result(result_shape, TifArray, predicts, RepetitiveLength, RowOver, ColumnOver)
writeTiff(ResultPath, result_data,im_geotrans,im_proj)
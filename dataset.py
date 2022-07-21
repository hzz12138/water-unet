import torch
import glob
import os
from osgeo import gdal
from torch.utils.data import Dataset


def read_Tiff(filename):
    """
    读取影像，返回影像矩阵
    """
    dataset = gdal.Open(filename)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    return data


class Data_Loader(Dataset):
    def __init__(self, data_path):
        """初始化Data_Loader"""
        self.data_path = data_path
        # 根据data_path，读取所有.tif文件
        self.images_path = glob.glob(os.path.join(data_path,'image', '*.tif'))

    def __getitem__(self, item):
        """利用生成器，返回image、label"""
        # 获取image、label的路径
        image_path = self.images_path[item]
        label_path = image_path.replace('image', 'label')
        # 读取image、label数据矩阵
        image = read_Tiff(image_path)
        label = read_Tiff(label_path)
        # 将label中255归一化到1
        label = label / 255
        # 返回image、label影像矩阵
        return image, label

    def __len__(self):
        """返回数据长度"""
        return len(self.images_path)


# if __name__ == "__main__":
#     data_path = 'train'
#     dataset = Data_Loader(data_path)
#     print(len(dataset))
#     train_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                batch_size=2,
#                                                shuffle=True)
#     i = 0
#     for image, label in train_loader:
#         print(image.shape,label.shape)
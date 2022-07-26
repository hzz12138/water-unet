import glob
import numpy as np
import torch
from UNet import Unet
from pre_process import ImageProcess

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = Unet(4, 1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('model/best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('111/image/*.tif')
    # 遍历所有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = (test_path.split('.')[0]).replace('image','result') + '.tif'
        # 读取图片
        # img = cv2.imread(test_path)
        image = ImageProcess(test_path)
        info = image.read_img_info()
        data = image.read_img_data()

        # 转为tensor
        img = np.expand_dims(data, 0)
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        image.write_img(save_res_path, pred, img_geotrans=info[3], img_proj=info[4])

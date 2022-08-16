import os.path

import predict_image
import pre_process
import glob
import time
import math
import torch
import numpy as np

if __name__ == '__main__':
    # 待处理影像目录
    # input_path = r'predict\input_files'
    input_path = r'predict\8bit_files'
    # 是否进行8bit转换
    trans_8bit = True
    # 是否输出8bit图像
    out_8bit = False
    out_8bit_path = r'predict\8bit_files'
    # 选用的模型 unet unetpp attu_net all
    model = 'unetpp'
    # 输出水体提取影像
    out_water_path = r'predict\watermask_files'

    # 遍历影像
    input_files = glob.glob(os.path.join(input_path, '*.tif'))
    #print(input_files)
    input_files.sort()
    print(input_files)

    init_time = time.perf_counter()
    # 程序开始
    for input_file in input_files:
        start = time.perf_counter()
        print(input_file, '影像处理开始')

        image_name = os.path.split(input_file)[-1]  # 获取影像文件名
        image = pre_process.ImageProcess(input_file)  # 读取影像
        info = image.read_img_info()  # 获取影像元信息
        print(type(info[4]), info[4])
        data = image.read_img_data()  # 获取影像数据

        if trans_8bit:
            data_8bit = image.trans_img_16bit_to_8bit()  # 将影像转为8bit

            if out_8bit:
                out_8bit_file = os.path.join(out_8bit_path, image_name.replace('.tif', '_8bit.tif'))  # 获取8bit影像文件名
                image.write_img(out_8bit_file, data_8bit, img_geotrans=info[3], img_proj=info[4])  # 写入8bit影像文件
        else:
            data_8bit = data
        del image

        # 水体提取
        data_8bit_swap = data_8bit.swapaxes(1, 0).swapaxes(1, 2)
        area_perc = 0.9
        RepetitiveLength = int((1 - math.sqrt(area_perc)) * 512 / 2)
        TifArray, RowOver, ColumnOver = predict_image.TifCroppingArray(data_8bit_swap, RepetitiveLength)
        DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if model == 'unet':
            model = predict_image.Unet(4, 1).cuda()
            model.to(DEVICE)
            model.load_state_dict(torch.load('model/best_model_unet0805.pth', map_location=DEVICE))
            out_water_file = os.path.join(out_water_path, image_name.replace('.tif', '_unet_water.tif'))  # 获取water影像文件名

        if model == 'unetpp':
            args = predict_image.getArgs()
            args.deepsupervision = False
            model = predict_image.NestedUNet(args, 4, 1).cuda()
            model.to(DEVICE)
            model.load_state_dict(torch.load('model/best_model_unetpp0808.pth', map_location=DEVICE))
            out_water_file = os.path.join(out_water_path,image_name.replace('.tif', '_unetpp_water.tif'))  # 获取water影像文件名

        if model == 'attu_net':
            model = predict_image.AttU_Net(4, 1).cuda()
            model.to(DEVICE)
            model.load_state_dict(torch.load('model/best_model_attu_net0808.pth', map_location=DEVICE))
            out_water_file = os.path.join(out_water_path,image_name.replace('.tif', 'attu_unet_water.tif'))  # 获取water影像文件名

        model.eval()
        predicts = []
        for i in range(len(TifArray)):
            for j in range(len(TifArray[0])):
                image = TifArray[i][j].swapaxes(0, 2).swapaxes(1, 2)
                img = np.expand_dims(image, 0)
                img_tensor = torch.from_numpy(img)
                img_tensor = img_tensor.to(device=DEVICE, dtype=torch.float32)
                pred = model(img_tensor)
                pred = np.array(pred.data.cpu()[0])[0]
                pred[pred >= 0.5] = 255
                pred[pred < 0.5] = 0
                predicts.append(pred)
            print(f"""{i}/{len(TifArray)}""")

        result_shape = (data_8bit_swap.shape[0], data_8bit_swap.shape[1])
        result_data = predict_image.Result(result_shape, TifArray, predicts, RepetitiveLength, RowOver, ColumnOver)

        predict_image.writeTiff(out_water_file, result_data, im_geotrans=info[3], im_proj=info[4])

        end = time.perf_counter()
        print(
            f"""{input_file},当前影像处理时间：{round((end - init_time) / 60, 2)}分钟,程序总运行时间：{round((end - start) / 60, 2)}分钟""")

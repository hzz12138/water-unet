from torch import optim
from dataset import Data_Loader
import torch
import torch.nn as nn
from UNet import Unet
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def acc_cal(predict_tensor, label_tensor):
    predict_data = np.squeeze(np.array(predict_tensor.data.cpu()), axis=1)
    label_data = np.squeeze(np.array(label_tensor.data.cpu()), axis=1)
    predict_data[predict_data >= 0.5] = 1
    predict_data[predict_data < 0.5] = 0
    return f1_score(label_data.flatten(), predict_data.flatten())


def train_net(net, device, train_data_path, val_data_path, batch_size, epochs, lr):
    # 读取所有测试样本，设定数据生成器
    train_dataset = Data_Loader(train_data_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Data_Loader(val_data_path)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # 定义RMSprop算法
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           amsgrad=False)
    # 定义Loss算法
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        print(f"""epoch:{epoch}/{epochs}""")
        # 按照batch_size开始训练
        i = 0
        base_lr = lr

        def adjust_lr(epoch):
            lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
            for params_group in optimizer.param_groups:
                params_group['lr'] = lr
            return lr

        # 训练模式
        total_train_loss = []
        total_train_acc = []
        net.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            i = i + 1
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            label = torch.unsqueeze(label, dim=1)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            acc = acc_cal(pred, label)
            loss = criterion(pred, label)
            total_train_acc.append(acc)
            total_train_loss.append(loss.item())

            if i % 10 == 0:
                print(
                    f"""epoch:{epoch}/{epochs},batch:{i}/{len(train_loader)},Loss/train:,{np.mean(total_train_loss)},lr:{optimizer.param_groups[0]['lr']},acc:{np.mean(total_train_acc)}""")
                total_train_loss = []
                total_train_acc = []

            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'model/best_model.pth')

            # 更新参数
            loss.backward()
            optimizer.step()
            adjust_lr(epoch)
            # 给torchvision添加数据
            writer.add_scalar('LOSS/Train_loss', float(loss), (epoch + 1))
            writer.add_scalar('ACC/Train_ACC', float(acc), (epoch + 1))

        if epoch % 5 == 0:
            torch.save(net.state_dict(), 'model/epoch_' + str(epoch) + '_model.pth')

        total_val_loss = []
        total_val_acc = []
        with torch.no_grad():
            for image, label in val_loader:
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                label = torch.unsqueeze(label, dim=1)
                # 使用网络参数，输出预测结果
                pred = net(image)
                # 计算loss
                acc = acc_cal(pred, label)
                loss = criterion(pred, label)
                total_val_acc.append(acc)
                total_val_loss.append(loss.item())

            print(f"""epoch:{epoch}/{epochs},Loss/train:,{np.mean(total_val_loss)},acc:{np.mean(total_val_acc)}""")

            # 给torchvision添加数据
            writer.add_scalar('LOSS/Validation_loss', float(np.mean(total_val_loss)), (epoch + 1))
            writer.add_scalar('ACC/Validation_ACC', float(np.mean(total_val_acc)), (epoch + 1))


if __name__ == "__main__":
    # 路径设定
    train_data_path = 'train'
    val_data_path = 'validation'
    # 超参数设定
    batch_size = 8
    epochs = 100
    lr = 1e-2
    step = [20, 40, 60, 80]
    device = 'cuda'
    # 定义Unet网络
    net = Unet(4, 1).cuda()
    summary(net, input_size=(4, 512, 512), device='cuda')
    writer = SummaryWriter('./runs')

    # net.load_state_dict(torch.load('model/best_model.pth'))
    # net.eval()
    train_net(net=net, device=device, train_data_path=train_data_path, val_data_path=val_data_path,
              batch_size=batch_size, epochs=epochs, lr=lr)
    writer.close()

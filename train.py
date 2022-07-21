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


def acc_cal(predict_tensor,label_tensor):
    predict_data = np.squeeze(np.array(predict_tensor.data.cpu()),axis=1)
    label_data = np.squeeze(np.array(label_tensor.data.cpu()), axis=1)
    predict_data[predict_data >= 0.5] = 1
    predict_data[predict_data < 0.5] = 0
    return f1_score(label_data.flatten(), predict_data.flatten())
    # score_list = []
    # for dim in range(predict_data.shape[0]):
    #     score_list.append(f1_score(label_data[dim].flatten(), predict_data[dim].flatten()))
    # # print(np.mean(score_list))
    # return np.mean(score_list)


def train_net(net, device, train_data_path, batch_size, epochs, lr):
    # 读取所有样本，设定数据生成器
    dataset = Data_Loader(train_data_path)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    # 定义RMSprop算法
    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           amsgrad=False)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        print(f"""epoch:{epoch}/{epochs}""")
        # 按照batch_size开始训练
        i = 0

        base_lr = lr
        def adjust_lr(epoch):
            lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
            for params_group in optimizer.param_groups:
                params_group['lr'] = lr
            return lr

        for image, label in train_loader:
            optimizer.zero_grad()
            i = i + 1
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            label = torch.unsqueeze(label, dim=1)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # pred = np.array(pred.data.cpu())[:,0,:,:]
            # 计算loss
            acc = acc_cal(pred,label)
            loss = criterion(pred, label)
            #print('Loss/train', loss.item())
            if i%10 == 0:
                print(f"""epoch:{epoch}/{epochs},batch:{i}/{len(train_loader)},Loss/train:,{loss.item()},lr:{optimizer.param_groups[0]['lr']},acc:{acc}""")
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
            adjust_lr(epoch)

            writer.add_scalar('LOSS/Train_loss', float(loss), (epoch + 1))


if __name__ == "__main__":
    # 路径设定
    train_data_path = 'train'
    # 超参数设定
    batch_size = 2
    epochs = 100
    lr = 1e-3
    step = [20, 40, 60, 80]
    device = 'cuda'
    # 定义Unet网络
    net = Unet(4,1).cuda()
    summary(net, input_size=(4, 512, 512), device='cuda')
    writer = SummaryWriter('./runs')

    net.load_state_dict(torch.load('best_model.pth'))
    # net.eval()
    train_net(net=net, device=device, train_data_path=train_data_path, batch_size=batch_size, epochs = epochs, lr = lr)
    writer.close()
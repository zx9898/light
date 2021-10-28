import os
from model.model_net import  *
import torch
import torch.nn as nn
from torch import optim
from dataset import *
from model.light_net import*

def train_net(model, device, img_path, epochs=50, batch_size=2, lr=1e-8):
    # load train datasets
    data_tarin = ISBI_Loader(img_path)
    train_loader = torch.utils.data.DataLoader(
        dataset=data_tarin,
        batch_size=batch_size,
        shuffle=True
    )
    # 定义RMSprop算法
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义BCELoss算法
    criterion = nn.BCEWithLogitsLoss()
    # 定义best_loss:
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        # 训练模式：
        for image, label in train_loader:
            optimizer.zero_grad()
            # 数据 copy到device
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = model(image)
            # 计算loss
            loss = criterion(pred, label)
            print('loss/train:', loss.item())
            # 保存最小的loss
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'best_model.pth')
                file = open('./loss_train.txt',"a")
                file.write('bestloss:'+str(best_loss)+"\n")
                file.close()
            # 更新参数
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = lightNet(in_channels=1, num_classes=1)
    model.to(device=device)
    img_path = 'data/train'
    label_path = 'data/train'
    train_net(model, device, img_path)
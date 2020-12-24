from unet_model import UNet
from utils.DALOADER import DealDataset
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import sys

class Logger():
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def train_net(net, device, data_path, epochs=300, batch_size=2, lr=0.0001):
    # 加载训练集
    name_dataset = DealDataset(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=name_dataset, batch_size=batch_size, shuffle=True) #shuffle 填True 就会打乱

    #定义算法
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3, last_epoch=-1)
    #定义Loss
    loss1 = nn.BCEWithLogitsLoss()
    loss2 = SoftDiceLoss()
    # best_loss统计，初始化为正无穷
    # best_loss = float('inf')
    # best_dice = -best_loss
    sys.stdout = Logger()
    for epoch in tqdm(range(epochs)):
        # 训练模式
        net.train()
        for batch in train_loader:
            optimizer.zero_grad()
            image = batch['image']
            label = batch['label']
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            # 计算loss
            c1 = loss1(pred, label)
            c2 = loss2(pred, label)
            loss = c1+c2
            # loss = c2
            pred = torch.sigmoid(pred)
            pred = (pred>0.5).float()
            print(pred.sum())
            dice = dice_coeff(pred, label)
            print('Dice/train', dice.item(),'\t','Loss/train', loss.item())
            # 保存loss值最小的网络参数
            # if dice > best_dice:
            #     best_dice = dice
            #     torch.save(net.state_dict(), 'best_model2021.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        torch.save(net.state_dict(), f'CP_epoch{epoch + 1}.pth')
        # scheduler.step()


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=3, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    # data_path = "../input/my-eye/data/CHASE/train"
    data_path = "./data/CHASE/train"
    train_net(net, device, data_path)

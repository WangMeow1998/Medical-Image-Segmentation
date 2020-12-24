import glob
import numpy as np
import torch
import os
from tqdm import tqdm
from unet_model import UNet
from PIL import Image
from utils.DALOADER import DealDataset

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=3, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('./save_model/CP_epoch300.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    data_path = "./data/CHASE/test"
    tests_path = glob.glob(os.path.join(data_path, r'image/*.jpg'))
    name_dataset = DealDataset(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=name_dataset, batch_size=1,shuffle=False)  # shuffle 填True 就会打乱
    # print(train_loader.len)
    # save_res_path = test_path.split('.')[0] + '_res2021.png'
    # 遍历所有图片
    total_batch = int(len(name_dataset)/1)
    bar = tqdm(enumerate(train_loader),total=total_batch)
    for batch_index, batch in bar:
        image = batch['image']
        # label = batch['label'].squeeze(0)
        path = batch['path']
        # print(path)
        save_res_path = '.' + path[0].split('.')[1] + '_res.png'
        # 保存结果地址
        # print(save_res_path)
        image = image.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(image)
        pred = torch.sigmoid(pred)
        pred = (pred>0.5).float()
        # dice = dice_coeff(pred, label)
        # print(dice.item())
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred = pred * 255
        # print(np.where(pred>=0))
        # pred[pred >= 0.5] = 255
        # pred[pred < 0.5] = 0
        # 保存图片
        #pred = cv2.resize(pred, (desired_size , desired_size ))
        #image = cv2.resize(new_im, (int(desired_size / 2), int(desired_size / 2)))
        # print(pred.shape)
        pred = Image.fromarray(pred).convert('RGB')
        pred.save(save_res_path)
        # cv2.imwrite(save_res_path, pred)
import torch
import numpy as np
import glob
import os
from PIL import Image


class DealDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, scale=0.6):
        # 1：所有图片和标签的路径
        self.images_path = glob.glob(os.path.join(data_path, r'image/*.jpg'))
        self.labels_path = glob.glob(os.path.join(data_path, r'label/*.png'))
        self.scale = scale

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, index):
        # 2：根据index取得相应的一幅图像，一幅标签的路径

        image_path = self.images_path[index]##self.images_path[item]是对应路径；通过image open得到图片
        label_path = self.labels_path[index]
        # print(image_path)
        # save_res_path = '.'+image_path.split('.')[1] + '_res.png'
        # print(save_res_path)
        #3.将图片和label读出。
        image = Image.open(image_path)
        # print(image.shape)
        label = Image.open(label_path).convert("L")
        #PIL.Image读取彩色图片：RGB， size:(w,h)，转成numpy后变成(h,w,c)
        image = self.preprocess(image,self.scale)
        label = self.preprocess(label,self.scale)
        # print(image.shape,label.shape)
        return {
            'image': torch.from_numpy(image).type(torch.FloatTensor),
            'label': torch.from_numpy(label).type(torch.FloatTensor),
            'path': image_path
        }

    def __len__(self):
        return len(self.images_path)



# if __name__ == '__main__':
#
#     data_transform = transforms.Compose([transforms.ToTensor()])
#     size = 256
#     # images_path = r"./data/CHASE/train/image"
#     # labels_path = r"./data/CHASE/train/label"
#     data_path = r"./data/CHASE/train"
#     name_dataset = DealDataset(data_path,size,transform =data_transform )
#     train_dataloader = torch.utils.data.DataLoader(dataset = name_dataset,batch_size = 16,shuffle = False) #shuffle 填True 就会打乱

import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DiseaseDataset(Dataset):
    def __init__(self, idx2imgname, name_list, transform=None):
        self.idx2imgname = idx2imgname
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        image_name = self.name_list[index]
        img = Image.open(image_name)
        label = np.array([classes for classes in self.idx2imgname if image_name in self.idx2imgname[classes]][0])

        if self.transform:
            img = self.transform(img)
        return img, torch.from_numpy(label)


def OurData(path_dir):
    # 用来存储图片的路径
    name_list = []
    train_list = []
    val_list = []

    idx = -1
    idx2imgname = {}
    idx2class =[]

    # path_dir = '.\DataSets'
    random.seed(123)

    for dirs in os.listdir(path_dir):
        dir_path = os.path.join(path_dir, dirs)
        if os.path.isdir(dir_path):
            idx += 1
            img_name = glob.glob(dir_path + '/*.jpg')
            idx2imgname[idx] = img_name
            idx2class.append(dirs)

            name_list += img_name
        else:
            pass

    # 随机打乱顺序
    name_list.sort()
    random.shuffle(name_list)
    length = len(name_list)

    with open('./alltrain.txt','w+',encoding='utf-8') as f:
        with open('./train_1.txt','w+',encoding='utf-8') as t:
            with open('./val_1.txt','w+',encoding='utf-8') as v:
                f.truncate()
                t.truncate()
                v.truncate()

    with open('./alltrain.txt','a+',encoding='utf-8') as f:
        for names in name_list:
            f.write(names+'\n')

    with open('./alltrain.txt','r+',encoding='utf-8') as f:
        with open('./train_1.txt','a+',encoding='utf-8') as t:
            with open('./val_1.txt','a+',encoding='utf-8') as v:

                x_1 = int(np.floor(0.1*length))
                x_2 = int(np.floor(0.2*length))
                x_3 = int(np.floor(0.3*length))
                x_4 = int(np.floor(0.4*length))
                X_5 = int(np.floor(0.5*length))
                X_6 = int(np.floor(0.6*length))
                X_7 = int(np.floor(0.7*length))
                X_8 = int(np.floor(0.8*length))
                X_9 = int(np.floor(0.9*length))

                train_list = name_list[:x_1] + name_list[x_2:]
                for m in train_list:
                    t.write(m+'\n')
                val_list = name_list[x_1:x_2]
                for n in val_list:
                    v.write(n+'\n')

    IMG_SIZE = (224, 224)
    # BATCH_SIZE = 64

    IMG_MEAN = [0.485, 0.456, 0.406]      # 0.507075, 0.486549, 0.440918,yuan:0.485, 0.456, 0.406
    IMG_STD = [0.229, 0.224, 0.225]      # 0.267334, 0.256438, 0.276150 yuan:0.229, 0.224, 0.225

    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        # transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    # 创建自建的Dataset
    TrainDataSet = DiseaseDataset(idx2imgname, train_list, train_transforms)
    ValDataSet = DiseaseDataset(idx2imgname, val_list, val_transforms)
    # 加载torch的数据加载
    # TrainDataLoader = DataLoader(TrainDataSet,
    #                              batch_size=BATCH_SIZE,
    #                              shuffle=True,
    #                              num_workers=0,
    #                              pin_memory=True,
    #                              drop_last=True
    #                              )

    # ValDataLoader = DataLoader(ValDataSet,
    #                            #batch_size=1,
    #                            batch_size=BATCH_SIZE,
    #                            shuffle=True,
    #                            num_workers=0,
    #                            pin_memory=True
    #                            )

    return TrainDataSet, ValDataSet
    # return TrainDataLoader, ValDataLoader


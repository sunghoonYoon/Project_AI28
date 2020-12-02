import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import skimage
import numpy as np
import json
import random
import glob
from PIL import Image,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class AI_Dataset(Dataset):
    def __init__(self,mode='train',transform=None):
        super(AI_Dataset, self).__init__()
        assert mode in ('train', 'val_refine')
        self.mode = mode
        self.impth = os.path.join("F:\\AI_Whole_dataset",mode) #Simulator 1,2 for train, 3 for val
        self.img= glob.glob(os.path.join(self.impth,"*.png"))
        self.len= len(self.img)
        self.transform = transform

    def __getitem__(self, idx):

        img_name = self.img[idx]

        basename= os.path.basename(img_name)
        if "normal" in basename:
            label = torch.FloatTensor([0,1])
        else:
            label = torch.FloatTensor([1,0])

        img = Image.open(img_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        label = label

        if self.mode=='train':
            return img,label
        else:
            return img,label,img_name

    def __len__(self):
        return self.len

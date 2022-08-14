import os.path as osp
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from PIL import Image
import random

random.seed(0)

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, 
                path, 
                time, 
                dt,
                intensity):
        self.path = path
        data = []
        label = []
        classes = os.listdir(path)
        print(classes)
        for class_folder in classes:
            folder_images_path = os.listdir(osp.join(path, class_folder))
            for image_path in folder_images_path:
                data.append(osp.join(path, class_folder, image_path))
                label.append(int(class_folder))

        self.data = data
        self.label = label
        self.encoding = PoissonEncoder(time=time, dt=dt)
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x * intensity)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path))
        return {
            'image': image,
            'encoded_image': self.encoding(image),
            'label': label,
        }


class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per + 1

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


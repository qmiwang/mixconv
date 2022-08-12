import os
import collections
import torch
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
import random
from .utils import *
import torchio as tio
from .ZNormalization import ZNormalization 
# One way:
import pickle
from itertools import islice
from .MyQueue import MyQueue

Record = collections.namedtuple('Record', 
                                ['image', 'label'])

          
#tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.15)
#tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5, p=0.5)               
                
class SegDataset(Dataset):
    def __init__(self, data_root, sample_txt, 
                 dataset_info,
                 samples_per_volume=1, 
                 patch_size=[128, 128, 64],
                 patch_overlap=[64, 64, 32],
                 num_workers=4,
                 batch_size=4,
                 num_iterations=250,
                 split="train", 
                 normalization='znorm',
                 trans=[],
                 **kwargs):
        super(SegDataset, self).__init__()
        print(normalization)
        

        self.trans = trans
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        with open(dataset_info, 'rb') as info_fl:
            self.info = pickle.load(info_fl)
            print(self.info)
        self.split = split
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.samples_per_volume = samples_per_volume
        for k in kwargs:
            self.__dict__[k] = kwargs[k]
        self.records = []
        if isinstance(sample_txt, list):
            for sample_path in sample_txt:
                self.records.append(Record(os.path.join(data_root, sample_path[0]), 
                                           os.path.join(data_root, sample_path[1])))
        else:
            with open(sample_txt, 'r') as f:
                for sample in f:
                    sample_path = sample.strip().split(',')
                    self.records.append(Record(os.path.join(data_root, sample_path[0]), 
                                               os.path.join(data_root, sample_path[1])))
        subjects = [tio.Subject(
                image=tio.ScalarImage(record.image),
                label=tio.LabelMap(record.label),
            ) for record in self.records]
        transform_list = []
        if normalization == 'znorm':
            from torchio.transforms import ZNormalization
            transform_list.append(ZNormalization())
        elif normalization == 'bnorm':
            transform_list.append(tio.Clamp(self.info["p_00_5"], self.info["p_99_5"]))
            from .ZNormalization import ZNormalization
            normalization = ZNormalization(mean_std=(self.info["mean"], self.info["std"]))
            transform_list.append(normalization)
        TRANS ={
            'affine': tio.RandomAffine(scales=(0.75, 1.25), degrees=(-30,30), isotropic=True, p=0.2),
            'noise': tio.RandomNoise(mean=0, std=0.1, p=0.15),
            'blur': tio.RandomBlur(std=(0.05, 0.15), p=0.2),
            'flip': tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5)
        }      
        if split == "train":
            print(patch_size)
            transform_list = transform_list + [TRANS[tran] for tran in trans]
            self.transforms = tio.Compose(transform_list)
            self.subjects_dataset = tio.SubjectsDataset(subjects, transform=self.transforms)
            self.sampler = tio.LabelSampler(patch_size, "label", {0:2,1:8})
            self.patches_queue = MyQueue(
                self.subjects_dataset,
                batch_size*min(samples_per_volume, 4),
                samples_per_volume,
                self.sampler,
                num_workers=num_workers,
            )
            self.patches_queue._fill()
        else:
            transform_list = transform_list + [TRANS[tran] for tran in trans]
            self.transforms = tio.Compose(transform_list)
            self.subjects_dataset = tio.SubjectsDataset(subjects, transform=self.transforms)
            

    def __getitem__(self, idx):
        if self.split == "train":
            #print(len(self.patches_queue.patches_list))
            return self.patches_queue[idx]
        elif self.split == "test" or self.split == "eval":
            return self.subjects_dataset[idx]

    def __len__(self):
        if self.split == "train":
            return self.num_iterations * self.batch_size
        elif self.split == "test" or self.split == "eval":
            return len(self.subjects_dataset)

from __future__ import print_function, division
import os, json, random
from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from envs.config import Config
from torchvision import transforms
import cv2
import utils.video_transforms as video_transforms
#import video_transforms as video_transforms

class URFDFusion(Dataset):
    NUM_CLASSES = 2
    def __init__(self,
                 dataset_dict,
                 base_dir=Config.get_dataset_path('urfdfusion'),
                 input_size=224,
                 stack_size=10,
                 split='train',
                 ):
        super().__init__()
        self.dataset_dict = dataset_dict
        self._base_dir = base_dir

        self.input_size = input_size
        self.stack_size = stack_size

        self.split = split

        self.videos = self.dataset_dict[self.split]['video_name']
        self.labels = self.dataset_dict[self.split]['labels']
        self.num_frames = self.dataset_dict[self.split]['num_frames']
        self.data = dataset_dict[self.split]['data']

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        rgb, flow = self.get_data(self.data[index])
        if self.split == 'train':
            rgb = self.transform_tr(rgb, is_flow=False)
            flow = self.transform_tr(flow, is_flow=True)

        elif self.split == 'val':
            rgb = self.transform_tr(rgb, is_flow=False)
            flow = self.transform_val(flow, is_flow=True)
       
        target = int(self.labels[index]) - 1

        return {'rgb': rgb, 'flow': flow, 'label': target}

    def transform_tr(self, sample, is_flow=False):
        if is_flow:
            composed_transforms = video_transforms.Compose([
                #video_transforms.MultiScaleCrop((224, 224), [1.0, 0.875, 0.75]),
                #video_transforms.RandomHorizontalFlip(),
                video_transforms.CenterCrop((224, 224)),
                video_transforms.ToTensor(),
                video_transforms.Normalize([0.5, 0.5] * self.stack_size, [0.226, 0.226] * self.stack_size),
            ])
        else:
            composed_transforms = video_transforms.Compose([
                #video_transforms.MultiScaleCrop((224, 224), [1.0, 0.875, 0.75]),
                #video_transforms.RandomHorizontalFlip(),
                video_transforms.CenterCrop((224, 224)),
                video_transforms.ToTensor(),
                video_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        return composed_transforms(sample)

    def transform_val(self, sample, is_flow=False):
        if is_flow:
            composed_transforms = video_transforms.Compose([
                video_transforms.CenterCrop((224, 224)),
                video_transforms.ToTensor(),
                video_transforms.Normalize([0.5, 0.5] * self.stack_size, [0.226, 0.226] * self.stack_size),
            ])
        else:
            composed_transforms = video_transforms.Compose([
                video_transforms.CenterCrop((224, 224)),
                video_transforms.ToTensor(),
                video_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
                
        return composed_transforms(sample)

    def get_data(self, data):
        dir_name = data.split(',')[0]
        frame_idx = int(data.split(',')[1])
        
        rgb = self.get_rgb(dir_name, frame_idx)
        flow = self.get_flows(dir_name, frame_idx)
        return rgb, flow

    def get_rgb(self, dir_name, frame_idx):
        frame_dir = os.path.join(self._base_dir, 'URFD_images', dir_name)
        rgb_img_path = os.path.join(frame_dir, 'rgb_{0:05d}.jpg'.format(frame_idx))

        assert(os.path.exists(rgb_img_path))

        rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)
        return rgb_img

    def get_flows(self, dir_name, frame_idx):
        flow = []
        frame_dir = os.path.join(self._base_dir, 'URFD_opticalflow', dir_name)
        for stack_idx in range(frame_idx, frame_idx + self.stack_size):
            x_img_path = os.path.join(frame_dir, 'flow_x_{0:05d}.jpg'.format(stack_idx))
            y_img_path = os.path.join(frame_dir, 'flow_y_{0:05d}.jpg'.format(stack_idx))
            
            assert(os.path.exists(x_img_path))
            assert(os.path.exists(y_img_path))
            
            x_img = cv2.imread(x_img_path, cv2.IMREAD_GRAYSCALE)
            y_img = cv2.imread(y_img_path, cv2.IMREAD_GRAYSCALE)
            
            flow.append(np.expand_dims(x_img, 2))
            flow.append(np.expand_dims(y_img, 2))

        flow = np.concatenate(flow, axis=2)
        return flow

    def __str__(self):
        return 'URFDFusion(split=' + self.split + ')'

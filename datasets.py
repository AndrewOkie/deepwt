import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import sobel
from scipy.ndimage.morphology import distance_transform_edt

import torch
import torch.nn.functional as F
import torch.utils.data as data


class DWTDataset(data.Dataset):

    def __init__(self, dataset_dir, split, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.split = split

        self.meta = pd.read_csv(os.path.join(self.dataset_dir, 'metadata.csv'))
        self.meta = self.meta[self.meta['split'] == self.split]

    def __len__(self):
        return len(self.meta)

    @staticmethod
    def gen_maps(masks):
        depth_map = np.zeros((masks.shape[0], masks.shape[1]), dtype=np.float32)
        gradient_map = np.zeros((2, masks.shape[0], masks.shape[1]), dtype=np.float32)
        weight_map = np.zeros((masks.shape[0], masks.shape[1]), dtype=np.float32)

        for id in np.unique(masks)[1:]:
            instance = 1 * (masks == id)

            if instance.sum() < 36:
                continue

            depth = distance_transform_edt(instance)
            gradient = np.array(np.gradient(depth, axis=(1, 0))) # xy gradient
            weight = instance * 200 / np.sqrt(instance.sum())

            depth_map += depth
            gradient_map += gradient
            weight_map += weight

        edges = np.hypot(sobel(masks, axis=0), sobel(masks, axis=1))
        gradient_map = gradient_map * (edges == 0)

        # quantize depth map
        depth_bins = [0, 1, 2, 3, 4, 5, 7, 9, 12, 15, 19, 24, 30, 37, 45, 54, np.inf]
        for i in range(len(depth_bins) - 1):
            depth_map[(depth_map > depth_bins[i]) & (depth_map <= depth_bins[i + 1])] = i

        return depth_map.astype(np.int64), gradient_map, weight_map

    def __getitem__(self, idx):
        image_id = self.meta.iloc[idx]['image_id']

        image = cv2.imread(os.path.join(self.dataset_dir, image_id, 'image.png'), cv2.IMREAD_UNCHANGED)
        masks = cv2.imread(os.path.join(self.dataset_dir, image_id, 'masks.png'), cv2.IMREAD_UNCHANGED)
        
        if self.transform:
            image, masks = self.transform(image, masks)

        depth_map, gradient_map, weight_map = self.gen_maps(masks.numpy())

        # normalize gradient map
        gradient_map = torch.from_numpy(gradient_map)
        gradient_map = F.normalize(gradient_map, dim=0)

        return image, (masks > 0).float(), torch.from_numpy(depth_map), gradient_map, torch.from_numpy(weight_map)

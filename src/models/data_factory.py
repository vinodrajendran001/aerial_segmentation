# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import transform as sk_transform
from skimage.io import imread
import os
from glob import glob


class Dataset(Dataset):    
    def __init__(self, split_file_image, split_file_label, using_onehot):

        self.split_file_image = sorted(glob(split_file_image + "/*.png"))
        self.split_file_label = sorted(glob(split_file_label + "/*.png"))
        self.using_onehot = using_onehot

    def __len__(self):
        return len(self.split_file_image)

    def __getitem__(self, idx):

        # Reading aerial image
        img_name = self.split_file_image[idx]
        image_single = imread(img_name)/ 255.0

        label_name = self.split_file_label[idx]
        label_single = imread(label_name)

        # splitting label

        # Converting labels to one-hot
        label_one_hot = 0.0 * image_single  

        # building channel 0
        label_one_hot[:, :, 0] = 1 * (
            np.logical_and(np.equal(label_single[:, :, 0], 255), np.equal(label_single[:, :, 2], 0)))

        # road channel 1
        label_one_hot[:, :, 1] = 1 * (
            np.logical_and(np.equal(label_single[:, :, 0], 0), np.equal(label_single[:, :, 2], 255)))

        # background, channel 2
        label_one_hot[:, :, 2] = 1 * (
            np.logical_and(np.equal(label_single[:, :, 0], 255), np.equal(label_single[:, :, 1], 255)))

        label_one_hot[:, :, 2] = 1 * np.logical_and(label_one_hot[:, :, 2], np.equal(label_single[:, :, 2], 255))

        '''
        # fixing some noisy, left-out pixels, assigning them to BG . These are the ones ==0 in all 3 channels
        all_zeros = np.logical_and(np.equal(label_one_hot[:, :, 0], 0), np.equal(label_one_hot[:, :, 1], 0))
        all_zeros = np.logical_and(all_zeros, np.equal(label_one_hot[:, :, 2], 0))

        # add these noisy pixels to background
        label_one_hot[:, :, 2] += 1 * all_zeros  
        '''


        if not self.using_onehot:
            label_one_hot = np.argmax(label_one_hot, 2)

        return image_single, label_one_hot


def get_dataset(mode, config):
    # Get dataset object by its name and mode (train/test)

    # set data directory
    data_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    

    if mode == 'train':
        split_file_image = os.path.join(data_folder, 'data', 'processed', mode,  'images') 
        split_file_label = os.path.join(data_folder, 'data', 'processed', mode,  'labels')
    elif mode == 'test':
        split_file_image = os.path.join(data_folder, 'data', 'processed', mode,  'images') 
        split_file_label = os.path.join(data_folder, 'data', 'processed', mode,  'labels')
    else:
        raise ValueError("Mode {} is unknown".format(mode))


    ds = Dataset(split_file_image, split_file_label, using_onehot= False)


    # preparing pytorch data loader
    ds_final = torch.utils.data.DataLoader(ds, batch_size=config['train']['batch_size'], shuffle=config['train']['shuffle'], num_workers=config['train']['num_workers'])

    return ds_final
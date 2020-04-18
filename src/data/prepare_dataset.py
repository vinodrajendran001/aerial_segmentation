# -*- coding: utf-8 -*-
import os
from glob import glob
import json
import random
import shutil

if __name__ == '__main__':
    
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    #prepare dataset

    with open(os.path.join(ROOT_DIR, 'config.json')) as json_file:
        config = json.load(json_file)

    train_test_split = config['data']['train_percentage']

    process_path = os.path.join(ROOT_DIR, 'data', 'processed')
    # train path
    image_train_path = os.path.join(process_path, 'train', 'images')
    label_train_path = os.path.join(process_path, 'train', 'labels')
    # test path
    image_test_path = os.path.join(process_path, 'test', 'images')
    label_test_path = os.path.join(process_path, 'test', 'labels')

    #create directories
    if not os.path.exists(image_train_path):
        os.makedirs(image_train_path)
    if not os.path.exists(label_train_path):
        os.makedirs(label_train_path)
    if not os.path.exists(image_test_path):
        os.makedirs(image_test_path)
    if not os.path.exists(label_test_path):
        os.makedirs(label_test_path)

    # check existence of interim dataset
    images_interim_path = os.path.join(ROOT_DIR, 'data', 'interim', 'images')
    labels_interim_path = os.path.join(ROOT_DIR, 'data', 'interim', 'labels')
    if not os.path.exists(images_interim_path) and not os.path.exists(labels_interim_path):
        print ("Please run the make_dataset script to process the dataset before run this script")
    else:
        dl_image_path = glob(images_interim_path + "/*.png")
        dl_label_path = glob(labels_interim_path + "/*.png")

        pairs = list(zip(dl_image_path, dl_label_path))

        split = len(dl_image_path) * (train_test_split/100)

        train_set = pairs[:int(split)]
        test_set = pairs[int(split):]

        for train_file in train_set:
            shutil.copy(train_file[0], image_train_path)
            shutil.copy(train_file[1], label_train_path)

        for test_file in test_set:
            shutil.copy(test_file[0], image_test_path)
            shutil.copy(test_file[1], label_test_path)

        print ("Train and Test set are prepared successfully")
        




    

    

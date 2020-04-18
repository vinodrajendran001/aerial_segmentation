# -*- coding: utf-8 -*-
import numpy as np
from skimage import transform as sk_transform
from skimage import img_as_float
from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import os
from glob import glob
import json
import dload # for windows user install ssl and add to windows system path https://slproweb.com/products/Win32OpenSSL.html to enable ssl
import cv2
import requests, zipfile, io

# overlay img and label
def overlay(image, mask, alpha=1.0):
    img = cv2.addWeighted(image, alpha, mask, 1.0 - alpha, 0)
    return img

# convert to color
def convert_to_color(label_one_hot, resize_dim):
    colorimg = np.zeros((resize_dim,resize_dim,3))
    for y in range(label_one_hot.shape[0]):
        for x in range(label_one_hot.shape[1]):
            if label_one_hot[y][x] == 0:
                colorimg[y][x][0]=255
                colorimg[y][x][1]=0
                colorimg[y][x][2]=0
            elif label_one_hot[y][x] == 1:
                colorimg[y][x][0]=0
                colorimg[y][x][1]=0
                colorimg[y][x][2]=255
            elif label_one_hot[y][x] == 2:
                colorimg[y][x][0]=255
                colorimg[y][x][1]=255
                colorimg[y][x][2]=255
    
    return colorimg

if __name__ == '__main__':

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Download and extract the dataset if not available in data/raw directory
    with open(os.path.join(ROOT_DIR, 'config.json')) as json_file:
        config = json.load(json_file)
        
    Ndataset = 0
    download_path = os.path.join(ROOT_DIR, 'data', 'raw')
    cities = config['data']['city']
    urls = config['data']['url']

    if len(urls) == len(cities):
        Ndataset = len(urls)
    else:
        print("Number of entries in city and url from config json did not match")

    for count in range(Ndataset):
        if not os.path.exists(download_path + "/" + cities[count].lower()):
            os.makedirs(download_path + "/" + cities[count].lower())
            # dload.save_unzip(urls[count],download_path)
            r = requests.get(urls[0])
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(download_path)
            print ("Downloaded the {} dataset".format(cities[count].lower()))
            print ("Downloaded the {} dataset".format(cities[count].lower()))
    
    #process dataset

    image_size_full = config['data']['image_size_full']
    process_path = os.path.join(ROOT_DIR, 'data', 'interim')
    image_path = os.path.join(process_path, 'images')
    label_path = os.path.join(process_path, 'labels')
    # label_path_color = os.path.join(process_path, 'labels_color') 
    overlay_path = os.path.join(download_path, 'overlay')

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    # if not os.path.exists(label_path_color):
    #     os.makedirs(label_path_color)

    for data in range(Ndataset):
        # get downloaded paths
        dl_image_path = glob(download_path + "/" + cities[data].lower() + "/*_image.png")
        dl_label_path = glob(download_path + "/" + cities[data].lower() + "/*_labels.png")

        if len(dl_image_path) != len(dl_label_path):
            print ("Number of images and labels for {} does not match".format(cities[data]))
        else:
            for idx in range(len(dl_image_path)):
                # original image
                img_big = imread(dl_image_path[idx])
                # cropping
                img_big = img_big[0:image_size_full[0], 0:image_size_full[1], :]  
                # original label
                label_big = imread(dl_label_path[idx])
                # cropping
                label_big = label_big[0:image_size_full[0], 0:image_size_full[1], :]
            
                if config['data']['overlay']:
                    if not os.path.exists(overlay_path):
                        os.makedirs(overlay_path)

                    overlay_img = overlay(img_big, label_big, alpha=0.7)
                    # save it to raw folder
                    fname_overlay = os.path.join(overlay_path,  os.path.basename(dl_image_path[idx])[:-9]  + "overlay" + ".png")

                    imsave(fname=fname_overlay, arr=overlay_img, check_contrast=False)


                # spliting image to 4x4 grid and resizing into 300x300 pixels 
                M = N = image_size_full[0]//config['data']['split']
                image_tiles = [img_big[x:x+M,y:y+N,:] for x in range(0,img_big.shape[0],M) for y in range(0,img_big.shape[1],N)]
                label_tiles = [label_big[x:x+M,y:y+N,:] for x in range(0,label_big.shape[0],M) for y in range(0,label_big.shape[1],N)]

                for tile in range(len(image_tiles)):
                    # Converting labels to one-hot
                    # 0: building red, 1 road blue, 2 BG white
                    label_one_hot = 0.0 * image_tiles[tile]  

                    # building channel 0
                    label_one_hot[:, :, 0] = 1 * (
                        np.logical_and(np.equal(label_tiles[tile][:, :, 0], 255), np.equal(label_tiles[tile][:, :, 2], 0)))

                    # road channel 1
                    label_one_hot[:, :, 1] = 1 * (
                        np.logical_and(np.equal(label_tiles[tile][:, :, 0], 0), np.equal(label_tiles[tile][:, :, 2], 255)))

                    # background, channel 2
                    label_one_hot[:, :, 2] = 1 * (
                        np.logical_and(np.equal(label_tiles[tile][:, :, 0], 255), np.equal(label_tiles[tile][:, :, 1], 255)))

                    label_one_hot[:, :, 2] = 1 * np.logical_and(label_one_hot[:, :, 2], np.equal(label_tiles[tile][:, :, 2], 255))

                    # fixing some noisy, left-out pixels, assigning them to BG . These are the ones ==0 in all 3 channels
                    all_zeros = np.logical_and(np.equal(label_one_hot[:, :, 0], 0), np.equal(label_one_hot[:, :, 1], 0))
                    all_zeros = np.logical_and(all_zeros, np.equal(label_one_hot[:, :, 2], 0))

                    # add these noisy pixels to background
                    label_one_hot[:, :, 2] += 1 * all_zeros   

                    # resizing
                    label_one_hot = sk_transform.resize(label_one_hot, (config['data']['image_resize'], config['data']['image_resize']), preserve_range=True)
                    img_single = sk_transform.resize(image_tiles[tile], (config['data']['image_resize'], config['data']['image_resize']), preserve_range=True)
                    

                    label_one_hot = np.argmax(label_one_hot, 2)

                    label_color = convert_to_color(label_one_hot, config['data']['image_resize'])

                    fname_images = os.path.join(image_path, os.path.basename(dl_image_path[idx])[:-4] + "_tile" + str(tile) + ".png")
                    fname_label = os.path.join(label_path, os.path.basename(dl_label_path[idx])[:-4] + "_tile" + str(tile) + ".png")
                    # fname_label_color = os.path.join(label_path_color, os.path.basename(dl_label_path[idx])[:-4] + "_tile" + str(tile) + ".png")

                    # save it to interim folder
                    imsave(fname=fname_images, arr=img_single, check_contrast=False)
                    # imsave(fname=fname_label, arr=label_one_hot, check_contrast=False)
                    imsave(fname=fname_label, arr=img_as_float(label_color), check_contrast=False)

        print ("Processed the {} dataset".format(cities[data].lower()))







        

        

        
        












        







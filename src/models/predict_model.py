import torch
import os
import sys 
from skimage import transform as sk_transform
from skimage.io import imread, imsave
import numpy as np
import torch.nn.functional as F
import json
from datetime import datetime
import matplotlib.pyplot as plt
from Unet import Net_lighter as Unet_class 
now = datetime.now()



# merge image
def merge_image(imaage_set, config):
    image_size_full = config['data']['image_size_full'][0]
    split = config['data']['split']
    step = image_size_full//split
    channel = 3
    merge = np.zeros((image_size_full,image_size_full,channel))
    i = 0
    for x in range(0, image_size_full, step):
        for y in range(0, image_size_full, step):
            if i == len(imaage_set):
                break
            merge[x:x + step, y:y + step, :] = imaage_set[i]
            i += 1

    return merge

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


def splitimage(image_path, config):
    
    image_size_full = config['data']['image_size_full']

    # Reading aerial image
    img_big = imread(image_path)/ 255.0

    # cropping
    img_big = img_big[0:image_size_full[0], 0:image_size_full[1], :]

    # spliting image to 4x4 grid and resizing into 300x300 pixels
    M = N = image_size_full[0] // config['data']['split']

    image_single_patch = [
        sk_transform.resize(
            img_big[x:x + M, y:y + N, :], (config['data']['image_resize'],
                                config['data']['image_resize']),
            preserve_range=True)
        for x in range(0, img_big.shape[0], M)
        for y in range(0, img_big.shape[1], N)
    ]
    
    return img_big, image_single_patch

def predict():

    # Read from command line
    model_checkpoint = sys.argv[1]
    image_path = sys.argv[2]
    out_file = sys.argv[3]

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    outdir = os.path.split(out_file)[0]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        

    with open(os.path.join(ROOT_DIR, 'config.json')) as json_file:
        config = json.load(json_file)

    segment_net = Unet_class()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loading segmentation net
    # segment_net = torch.load(model_checkpoint)
    segment_net.load_state_dict(torch.load(model_checkpoint))
    segment_net.eval()
    segment_net.to(device)

    print('Network loaded...')
    print(config)

   
    crop_image, images_single_patch = splitimage(image_path, config)

    subdivs = list()

    with torch.no_grad():
        print ("splitting into small tiles...")
        for tile in range(len(images_single_patch)):
            image = images_single_patch[tile]
            image = torch.from_numpy(image).float().to(device)
            image = image.unsqueeze(0)
            predicted = segment_net(image)
            # plt.imshow(np.transpose(predicted[0].cpu().detach().numpy(), (1, 2, 0)))
            predicted = torch.argmax(predicted[:, :, :, :], dim=1)
            predicted_color = convert_to_color(predicted[0], config['data']['image_resize'])
            resize = config['data']['image_size_full'][0] // config['data']['split']
            predicted_color = sk_transform.resize(predicted_color, (resize, resize), preserve_range=True)
            # plt.imshow(predicted_color)
            # plt.show()
            subdivs.append(predicted_color)

        print("processed all tiles....")

    predicted_image = merge_image(subdivs, config)

    imsave(fname=out_file, arr=predicted_image, check_contrast=False)

    print("Prediction completed...check your outfile")

    # return crop_image, predicted_image



if __name__ == "__main__":

    predict()
    '''
    model_checkpoint = "C:\\Users\\Vinod\\Documents\\audi_ml_challenge\\models\\20200415-152052\\trained_model_end.pth"
    image_path = "C:\\Users\\Vinod\\Documents\\audi_ml_challenge\\data\\raw\\berlin\\berlin5_image.png"
    out_file = "C:\\Users\\Vinod\\Documents\\audi_ml_challenge\\reports\\figures\\predict.png"

    crop_image, predicted_image = predict(model_checkpoint, image_path, out_file)
    '''

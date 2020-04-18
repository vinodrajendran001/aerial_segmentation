# This file loads the trained model from disk and evaluates the trained model TEN times and computes the average results.

import torch
import os
from data_factory import get_dataset
import numpy as np
import torch.nn.functional as F
from metrics import IoU
import json
from glob import glob
from datetime import datetime
from Unet import Net_lighter as Unet_class 
now = datetime.now()

# Compute evaluation this many times and then compute average
repeat_times = 10    


def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with open(os.path.join(ROOT_DIR, 'config.json')) as json_file:
        config = json.load(json_file)

    out_dir = os.path.join(ROOT_DIR, 'models')

    if not os.path.exists(out_dir):
        raise ValueError('The folder does not exist. Make sure the training has been completed before running this script')

    segment_net = Unet_class()

    latest_model = max(glob(out_dir + "/*"))
    trained_model = glob(os.path.join(out_dir, latest_model) + "/*.pth")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loading segmentation net
    if len(trained_model) == 2:
        # segment_net = torch.load(os.path.join(out_dir, latest_model, trained_model[1]))
        segment_net.load_state_dict(torch.load(os.path.join(out_dir, latest_model, trained_model[1])))
    else:
        # segment_net = torch.load(os.path.join(out_dir, latest_model, trained_model[0]))
        segment_net.load_state_dict(torch.load(os.path.join(out_dir, latest_model, trained_model[0])))

    segment_net.eval()
    segment_net.to(device)

    print('Network loaded...')
    print(config)

    ## getting the dataset
    mode = 'test'                 

    ds_test = get_dataset(mode, config)
    print('Data loaders have been prepared!')

    # Initialize metrics

    iou_build = 0
    iou_road = 0
    iou_bg = 0
    mIoU = 0
    fwIou = 0
    acc = 0


    with torch.no_grad():
        for t in range(repeat_times):  # evaluate everything 10 times

            for i, data in enumerate(ds_test, 0):
                images = data[0].type('torch.FloatTensor') # reading images

                # labels
                labels = data[1].type('torch.LongTensor')

                # segmentation performance
                predicted = segment_net(images)
                i1, i2, i3, i4, i5, i6 = IoU(predicted, labels, extra=True)
                iou_build += i1
                iou_road += i2
                iou_bg += i3
                mIoU += i4
                fwIou += i5
                acc += i6

            print('Completed ' + str(t) + 'out of ' + str(repeat_times))


    # average of segmentation numbers
    iou_build /= ( len(ds_test)  * repeat_times)
    iou_road /= (len(ds_test)  * repeat_times)
    iou_bg /= (len(ds_test) * repeat_times)
    mIoU /= (len(ds_test)  * repeat_times)
    fwIou /= (len(ds_test)  * repeat_times)

    acc /= (len(ds_test)  * repeat_times)

    print('Building IoU on test set = ' + str(iou_build))
    print('Road IoU on test set = ' + str(iou_road))
    print('BG IoU on test set = ' + str(iou_bg))
    print('mIoU on test set = ' + str(mIoU))
    print('Frequency weighted IoU on test set = ' + str(fwIou))
    print('Pixel accuracy on test set = ' + str(acc))

    fname =  os.path.join(ROOT_DIR, 'reports', now.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(fname):
        os.makedirs(fname)
    fname = os.path.join(fname, 'eval_results.txt')

    # saving results on disk
    with open(fname, 'w') as result_file:
        result_file.write('Logging... \n')
        result_file.write('\nBuilding IoU on test set =   ')
        result_file.write(str(iou_build))
        result_file.write('\nRoad IoU on test set =   ')
        result_file.write(str(iou_road))
        result_file.write('\nBG IoU on test set =   ')
        result_file.write(str(iou_bg))
        result_file.write('\nMean IoU on test set =   ')
        result_file.write(str(mIoU))
        result_file.write('\nfrequency weighted IoU on test set =   ')
        result_file.write(str(fwIou))
        result_file.write('\nPixel accuracy on test set =   ')
        result_file.write(str(acc))

    print('All done. Results saved in reports directory')

if __name__ == '__main__':
    main()
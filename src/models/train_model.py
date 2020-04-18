# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from data_factory import get_dataset
import numpy as np
from datetime import datetime
import os
from Unet import Net_lighter as Unet_class   # lighter U-Net qaurter the channels
from metrics import IoU
import json
now = datetime.now()

def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with open(os.path.join(ROOT_DIR, 'config.json')) as json_file:
        config = json.load(json_file)

    out_dir = os.path.join(ROOT_DIR, 'models', now.strftime("%Y%m%d-%H%M%S"))

    # setting output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print('Folder already exists. overwriting the results......')


    print('Configuration:')
    print(config['train'])

    ## Data loaders
    mode = 'train' # Training data loader
    ds_train = get_dataset(mode, config)

    mode = 'test' # validation data loader
    ds_test = get_dataset(mode, config)
    print('Data loaders have been prepared!')

    ## Model
    mode = 'train'
    net = Unet_class()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    print('Network loaded. Starting training...')

    # weights for  # building, road, BG - To balance the classes
    my_weight = torch.from_numpy(np.asarray([1, 2, 0.5])).type('torch.FloatTensor')
    criterion = torch.nn.CrossEntropyLoss(weight=my_weight)

    optim = torch.optim.Adam(net.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['learning_rate_decay'])

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    ep = config['train']['num_epochs'] # number of epochs

    # loss logs
    loss_train = 9999.0*np.ones(ep)
    temp_train_loss = 0
    loss_val = 9999.0*np.ones(ep)

    # training the network
    for epoch in range(ep):
        running_loss = 0.0
        running_ctr = 0

        # switch model to training mode, clear gradient accumulators
        net.train()

        # scheduler.step()  # update learning rate
        
        optim.zero_grad()

        t1 = datetime.now()

        for i, data in enumerate(ds_train, 0):
            optim.zero_grad()

            # reading images
            images = data[0].type('torch.FloatTensor')
            # labels
            labels = data[1].type('torch.LongTensor')

            predicted = net(images)

            loss = criterion(predicted, labels)

            loss.backward()

            optim.step()


            # print statistics
            running_loss += loss.item()
            running_ctr += 1
            if i %25 ==0:
                t2 = datetime.now()
                delta = t2 - t1
                t_print = delta.total_seconds()
                temp_train_loss = running_loss/25.0
                print('[%d, %5d out of %5d] loss: %f, time = %f' %
                      (epoch + 1, i + 1, len(ds_train) , running_loss / running_ctr,  t_print ))

                iou_build, iou_road, iou_bg = IoU(predicted, labels)
                print('building IoU = ' + str(iou_build) + ', road IoU = ' + str(iou_road) + ', background IoU = ' + str(iou_bg) )

                running_loss = 0.0
                running_ctr = 0
                t1 = t2

        net.eval()
        scheduler.step()  # update learning rate
        val_loss = 0

        with torch.no_grad():
            for i, data in enumerate(ds_test, 0):
                # reading images
                images = data[0].type('torch.FloatTensor')
                # labels
                labels = data[1].type('torch.LongTensor')

                predicted = net(images)

                loss = criterion(predicted, labels)

                # Val loss
                val_loss +=  loss.item()


            # print statistics
            val_loss = val_loss /len(ds_test)
            print('End of epoch ' + str(epoch + 1) + '. Val loss is ' + str(val_loss))

            print('Following stats are only for the last batch of the test set:')
            iou_build, iou_road, iou_bg = IoU(predicted, labels)
            print('building IoU = ' + str(iou_build) + ', road IoU = ' + str(iou_road) + ', background IoU = ' + str(
                iou_bg))

            # Model check point
            if val_loss < np.min(loss_val, axis=0):
               model_path = os.path.join(out_dir, "trained_model_checkpoint.pth")
            #    torch.save(net, model_path)
               torch.save(net.state_dict(), model_path)
               print('Model saved at epoch ' + str(epoch+1))


            # saving losses
            loss_val[epoch] = val_loss
            loss_train[epoch] = temp_train_loss

            temp_train_loss = 0  # setting additive losses to zero



    print('Training finished')
    # saving model
    model_path = os.path.join(out_dir, "trained_model_end.pth")
    # torch.save(net, model_path)
    torch.save(net.state_dict(), model_path)

    print('Model saved')

    # Saving logs in a text file in the output directory
    log_name = os.path.join(out_dir, "logging.txt")
    with open(log_name, 'w') as result_file:
        result_file.write('Logging... \n')
        result_file.write('Validation loss ')
        result_file.write(str(loss_val))
        result_file.write('\nTraining loss  ')
        result_file.write(str(loss_train))

    # saving loss curves
    a = loss_val
    b = loss_train

    plt.figure()
    plt.plot(b)
    plt.plot(a)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training loss', 'Validation Loss'])
    fname1 = str('loss.png')
    plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight')

    print('Training finished!!!')

if __name__ == '__main__':
    main()
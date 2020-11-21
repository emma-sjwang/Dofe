#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from dataloaders import utils
# from scipy.misc import imsave
from utils.Utils import joint_val_image, postprocessing, save_per_img
from utils.metrics import *
from datetime import datetime
import pytz
from networks.deeplabv3 import *
import cv2
import numpy as np
from medpy.metric import binary

def construct_color_img(prob_per_slice):
    shape = prob_per_slice.shape
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = prob_per_slice * 255
    img[:, :, 1] = prob_per_slice * 255
    img[:, :, 2] = prob_per_slice * 255

    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return im_color


def normalize_ent(ent):
    '''
    Normalizate ent to 0 - 1
    :param ent:
    :return:
    '''
    max = np.amax(ent)
    # print(max)

    min = np.amin(ent)
    # print(min)
    return (ent - min) / 0.4


def draw_ent(prediction, save_root, name):
    '''
    Draw the entropy information for each img and save them to the save path
    :param prediction: [2, h, w] numpy
    :param save_path: string including img name
    :return: None
    '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    # save_path = os.path.join(save_root, img_name[0])
    smooth = 1e-8
    cup = prediction[0]
    disc = prediction[1]
    cup_ent = - cup * np.log(cup + smooth)
    disc_ent = - disc * np.log(disc + smooth)
    cup_ent = normalize_ent(cup_ent)
    disc_ent = normalize_ent(disc_ent)
    disc = construct_color_img(disc_ent)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup_ent)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)


def draw_mask(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    cup = prediction[0]
    disc = prediction[1]

    disc = construct_color_img(disc)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)



def draw_boundary(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'boundary')):
        os.makedirs(os.path.join(save_root, 'boundary'))
    boundary = prediction[0]

    boundary = construct_color_img(boundary)
    cv2.imwrite(os.path.join(save_root, 'boundary', name.split('.')[0]) + '.png', boundary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='./logs/test1/20190506_221021.177567/checkpoint_200.pth.tar', help='Model path')
    parser.add_argument('--datasetTest', type=list, default=[1], help='test folder id contain images ROIs to test')
    parser.add_argument('--dataset', type=str, default='test', help='test folder id contain images ROIs to test')
    parser.add_argument('-g', '--gpu', type=int, default=0)

    parser.add_argument('--data-dir', default='../../../../Dataset/Fundus/', help='data root path')
    parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+',)
    parser.add_argument('--sync-bn', type=bool, default=False, help='sync-bn in deeplabv3+')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='freeze batch normalization of deeplabv3+')
    parser.add_argument('--movingbn', type=bool, default=False, help='moving batch normalization of deeplabv3+ in the test phase',)
    parser.add_argument('--test-prediction-save-path', type=str, default='./results/rebuttle-0401/', help='Path root for test image and mask')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file
    output_path = os.path.join(args.test_prediction_save_path, 'test' + str(args.datasetTest[0]), args.model_file.split('/')[-2])

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                    transform=composed_transforms_test, state='prediction')
    batch_size = 12
    test_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=1)

    # 2. model
    model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                    sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    # model_data = torch.load(model_file)

    checkpoint = torch.load(model_file)
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    if args.movingbn:
        model.train()
    else:
        model.eval()

    val_cup_dice = 0.0
    val_disc_dice = 0.0
    total_hd_OC = 0.0
    total_hd_OD = 0.0
    total_asd_OC = 0.0
    total_asd_OD = 0.0
    timestamp_start = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    total_num = 0
    OC = []
    OD = []

    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader),ncols=80, leave=False):
        data = sample['image']
        target = sample['label']
        img_name = sample['img_name']
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        prediction, dc, sel, _ = model(data)
        prediction = torch.nn.functional.interpolate(prediction, size=(target.size()[2], target.size()[3]), mode="bilinear")
        data = torch.nn.functional.interpolate(data, size=(target.size()[2], target.size()[3]), mode="bilinear")

        target_numpy = target.data.cpu()
        imgs = data.data.cpu()
        hd_OC = 100
        asd_OC = 100
        hd_OD = 100
        asd_OD = 100
        for i in range(prediction.shape[0]):
            prediction_post = postprocessing(prediction[i], dataset=args.dataset)
            cup_dice, disc_dice = dice_coeff_2label(prediction_post, target[i])
            OC.append(cup_dice)
            OD.append(disc_dice)
            if np.sum(prediction_post[0, ...]) < 1e-4:
                hd_OC = 100
                asd_OC = 100
            else:
                hd_OC = binary.hd95(np.asarray(prediction_post[0, ...], dtype=np.bool),
                                    np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
                asd_OC = binary.asd(np.asarray(prediction_post[0, ...], dtype=np.bool),
                                    np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
            if np.sum(prediction_post[0, ...]) < 1e-4:
                hd_OD = 100
                asd_OD = 100
            else:
                hd_OD = binary.hd95(np.asarray(prediction_post[1, ...], dtype=np.bool),
                                    np.asarray(target_numpy[i, 1, ...], dtype=np.bool))

                asd_OD = binary.asd(np.asarray(prediction_post[1, ...], dtype=np.bool),
                                    np.asarray(target_numpy[i, 1, ...], dtype=np.bool))
            val_cup_dice += cup_dice
            val_disc_dice += disc_dice
            total_hd_OC += hd_OC
            total_hd_OD += hd_OD
            total_asd_OC += asd_OC
            total_asd_OD += asd_OD
            total_num += 1
            for img, lt, lp in zip([imgs[i]], [target_numpy[i]], [prediction_post]):
                img, lt = utils.untransform(img, lt)
                save_per_img(img.numpy().transpose(1, 2, 0),
                             output_path,
                             img_name[i],
                             lp, lt, mask_path=None, ext="bmp")

    print('OC:', OC)
    print('OD:', OD)
    import csv
    with open('Dice_results.csv', 'a+') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        for index in range(len(OC)):
            wr.writerow([OC[index], OD[index]])

    val_cup_dice /= total_num
    val_disc_dice /= total_num
    total_hd_OC /= total_num
    total_asd_OC /= total_num
    total_hd_OD /= total_num
    total_asd_OD /= total_num

    print('''\n==>val_cup_dice : {0}'''.format(val_cup_dice))
    print('''\n==>val_disc_dice : {0}'''.format(val_disc_dice))
    print('''\n==>average_hd_OC : {0}'''.format(total_hd_OC))
    print('''\n==>average_hd_OD : {0}'''.format(total_hd_OD))
    print('''\n==>ave_asd_OC : {0}'''.format(total_asd_OC))
    print('''\n==>average_asd_OD : {0}'''.format(total_asd_OD))
    with open(osp.join(output_path, '../test' + str(args.datasetTest[0]) + '_log.csv'), 'a') as f:
        elapsed_time = (
                datetime.now(pytz.timezone('Asia/Hong_Kong')) -
                timestamp_start).total_seconds()
        log = [['batch-size: '] + [batch_size] + [args.model_file] + ['cup dice coefficence: '] + \
               [val_cup_dice] + ['disc dice coefficence: '] + \
               [val_disc_dice] + ['average_hd_OC: '] + \
               [total_hd_OC] + ['average_hd_OD: '] + \
               [total_hd_OD] + ['ave_asd_OC: '] + \
               [total_asd_OC] + ['average_asd_OD: '] + \
               [total_asd_OD] + [elapsed_time]]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    main()

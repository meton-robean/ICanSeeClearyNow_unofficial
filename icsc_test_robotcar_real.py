# PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision

import numpy as np
import cv2
import random
import time
import os
import argparse

import h5py
import skimage
import cv2

from skimage.measure import compare_psnr, compare_ssim
from model.discriminator import NLayerDiscriminator
from model.generator import  Derain_GlobalGenerator
from options import TrainOptions



def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)


def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        return data

def image_to_tensor(image):
    image = np.array(image, dtype='float32') / 255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    tensor = Variable(image).cuda()
    return tensor

def tensor_to_image(tensor):
    out = tensor.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    image = out[0, :, :, :] * 255.
    image = np.array(image, dtype='uint8')
    return image

def cut_imgae(path, img_size):
    input_image = cv2.imread(path)[:, :, :]
    h = round(input_image.shape[0] / img_size)
    w = round(input_image.shape[1] / img_size)
    sub_image_set = []
    for i in range(h):
        set = []
        for j in range(w):
            if (j != w - 1):
                set.append(input_image[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :])
            else:
                set.append(input_image[i * img_size:(i + 1) * img_size, j * img_size:, :])
        sub_image_set.append(set)
    return sub_image_set,h,w


def merge_image(sub_image_set,h,w):
    merge_img_set = []
    for i in range(h):
        for j in range(w):
            if (j == 0):
                merge_img = sub_image_set[i][j]
            else:
                merge_img = np.concatenate((merge_img, sub_image_set[i][j]), axis=1)
        merge_img_set.append(merge_img)

    for i in range(len(merge_img_set)):
        print(merge_img_set[i].shape)
        if (i == 0):
            merge_img = merge_img_set[i]
        else:
            merge_img = np.concatenate((merge_img, merge_img_set[i]), axis=0)
    return merge_img



def cut_batch_image_to_tensor(rain_image, cut_num):

    (H, W, C) = rain_image.shape
    cut_unit = W//cut_num
    sub_list = []
    for i in range(cut_num):
        
        sub_image = rain_image[:, i*cut_unit: cut_unit*(i+1), :]
        sub_image = np.array(sub_image, dtype='float32') / 255.
        sub_image = sub_image.transpose((2, 0, 1))
        sub_image = sub_image[np.newaxis, :, :, :]
        sub_image = torch.from_numpy(sub_image)
        sub_tensor= Variable(sub_image).cuda()
        sub_list.append(sub_tensor)
    batch_tensor = torch.cat(sub_list)
    print(batch_tensor.shape)
    return batch_tensor


def merge_batch_tensor_to_image(batch_tensor):
    img_list =[]
    for i in range(batch_tensor.shape[0]):
        img = batch_tensor[i].cpu().detach().numpy()
        img = img.transpose((1,2,0))
        image = img[ :, :, :] * 255.
        image = np.array(image, dtype='uint8')
        img_list.append(image)
    merge_img = np.concatenate(img_list, 1)
    return merge_img
    


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RobotCar_real_test")
    parser.add_argument("--clean_data_path", default="./data/robotcar_derain_segment/labelled/")
    parser.add_argument("--rain_data_path", default="./data/robotcar_derain_segment/labelled/")
    parser.add_argument("--Data_path", default="./data/")
    
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--save_path", type=str, default="./test/cityscapes/test/", help='path to save results')
    parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
    args = parser.parse_args()
    opt = TrainOptions().parse()


    from model.generator import Derain_GlobalGenerator
    from model.discriminator import Discriminator_n_layers

    ###### load model ######
    model = Derain_GlobalGenerator(input_nc=3, output_nc=3, ngf=16, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect').cuda()

    D = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=5, 
                                        norm_layer=nn.BatchNorm2d, 
                                        use_sigmoid=True, getIntermFeat=True).cuda()

    ##load trained model
    model.load_state_dict(torch.load('./Exp1_RobotCarReal-deraindrop/saved_models/generator_xxx.pth'))
    D.load_state_dict(torch.load('./Exp1_RobotCarReal-deraindrop/saved_models/discriminator_xxx.pth'))
    model.eval()
    D.eval()
    #####

    print('Loading robocar real dataset for test ...\n')

    Test_rain_image_name = read_data(args.Data_path + "Test_Rainy_image_name.h5")
    input_name_list = Test_rain_image_name
    input_path = args.rain_data_path

    Test_clean_image_name = read_data(args.Data_path + "Test_Clean_image_name.h5")
    gt_name_list = Test_clean_image_name
    gt_path = args.clean_data_path


    print("testing...")
    cumulative_psnr = 0
    cumulative_ssim = 0
    num = len(input_name_list)
    print('{} samples for testing....'.format(num))
    
    
    #### one pass a img ####
    final_derain_psnr =0; final_derain_ssim = 0;
    psnr = 0; ssim =0;
    for index in range(len(input_name_list)):
        print('processing ' + input_name_list[index].decode())
        print("ground_truth "+ gt_name_list[index].decode())
        gt_path = args.clean_data_path
        data_path = args.rain_data_path 
        
        gt_image = cv2.imread(gt_path + gt_name_list[index].decode())
        rain_image = cv2.imread(data_path +input_name_list[index].decode())
        #print('gt shape :', gt_image.shape)
        
        rain_image_tensor = image_to_tensor(rain_image)
        out = model(rain_image_tensor)
        #out_image = tensor_to_image(out)
        out_image = out.data.cpu().numpy()[0]
        out_image[out_image>1] = 1
        out_image[out_image<0] = 0
        out_image*= 255
        out_image = out_image.astype(np.uint8)
        out_image = out_image.transpose((1,2,0))
        print(out_image.shape)
        
        input_image = cv2.imread(data_path +input_name_list[index].decode())
        print('save pics ....')
        cv2.imwrite('./robotcar_real_test/clean/{}.png'.format(index), gt_image)
        cv2.imwrite('./robotcar_real_test/output/{}.png'.format(index), out_image)
        cv2.imwrite('./robotcar_real_test/input/{}.png'.format(index), input_image)
        psnr = calc_psnr( out_image ,gt_image)
        final_derain_psnr += psnr
        ssim = calc_ssim(out_image ,gt_image)
        final_derain_ssim +=ssim
        print('pic{} :'.format(index), 'psnr:{} '.format(psnr), 'ssim:{} '.format(ssim))
    print('*****final PSNR: {}'.format(final_derain_psnr/num))
    print('*****final SSIM: {}'.format(final_derain_ssim/num))



        



 




import argparse
import os
import numpy as np
import time
import datetime
import sys

import torch
from torch.autograd import Variable

#from pix2pix_models import Create_nets
from datasets import Get_dataloader
from options import TrainOptions
from optimizer import *
from utils import sample_images , LambdaLR

from model.discriminator import create_disc_nets, Disc_MultiS_Scale_Loss
from model.generator import  create_gen_nets
from model.vgg_loss import VGGLoss
from datasets import RobotCar_Real_Dataset
from torch.utils.data import DataLoader
import h5py


# Loss functions
def Get_loss_func(args):
    criterion_GAN = torch.nn.MSELoss()        
    criterion_pixelwise = torch.nn.L1Loss()  
    if torch.cuda.is_available():
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
    return criterion_GAN, criterion_pixelwise



#load the args
args = TrainOptions().parse()


# Initialize generator and discriminator
generator = create_gen_nets(args)
discriminator = create_disc_nets(args)
# if torch.cuda.device_count() > 1:
#     print('multi GPUS for training......')
#     discriminator = torch.nn.DataParallel(discriminator, device_ids=[0,1])
#     generator = torch.nn.DataParallel(generator, device_ids=[0,1])
# load model weight

# Loss functions
criterion_GAN, criterion_pixelwise = Get_loss_func(args)
criterion_Vgg = VGGLoss()
criterion_DiscMultiScaleLoss = Disc_MultiS_Scale_Loss()
# Optimizers
optimizer_G, optimizer_D = Get_optimizers(args, generator, discriminator)

# Configure dataloaders
# h5 file contains the path of imgs for training 
# you can replace the code of dataset processing here with the own code
print('Loading robotcar real dataset ...\n')
inpt_path = './data/robotcar_derain_segment/labelled/'
label_path  ='./data/robotcar_derain_segment/labelled/'
rain_path  ='./data/Train_Rainy_image_name.h5'
gt_path ='./data/Train_Clean_image_name.h5'

#train dataset
dataset_train = RobotCar_Real_Dataset(label_path, gt_path, inpt_path, rain_path, image_size=384)
train_dataloader = DataLoader(dataset=dataset_train, num_workers=4, batch_size=args.batch_size, shuffle=True)


print("# of training samples: %d\n" % int(len(dataset_train)))

#test dataset
test_rain_path = './data/Test_Rainy_image_name.h5'
test_gt_path = './data/Test_Clean_image_name.h5'
dataset_test = RobotCar_Real_Dataset(label_path, test_gt_path, inpt_path, test_rain_path, image_size=256)  
test_dataloader = DataLoader(dataset=dataset_test, num_workers=4, batch_size=10, shuffle=True)
print("# of test samples: %d\n" % int(len(test_dataloader)))
######


# ----------
#  Training
# ----------
from tensorboardX import SummaryWriter
writer = SummaryWriter('./robotcar_real_logs/')
count = 0


prev_time = time.time()

#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
for epoch in range(args.epoch_start, args.epoch_num):
    for i, batch in enumerate(train_dataloader):
        count+=1
        # Model inputs  realA is rain img,   realB is gt img 
        real_A, real_B = batch    
        real_A = Variable(real_A.cuda())
        real_B = Variable(real_B.cuda())
        #print('shape:', real_A.shape, real_B.shape)
        if i==0:
            # Calculate output of image discriminator (PatchGAN)
            D_out_size = 512//(2**args.n_D_layers) - 2
            D_out_size2 = 512//(2**args.n_D_layers) - 2
            print('patchGan Size:', D_out_size, D_out_size2)
            patch = (1, D_out_size, D_out_size2)
            
        # Adversarial ground truths
        #valid = Variable(torch.FloatTensor(np.ones((real_A.size(0), *patch))).cuda(), requires_grad=False)
        #fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), *patch))).cuda(), requires_grad=False)

        # Update learning rates
        #lr_scheduler_G.step(epoch)
        #lr_scheduler_D.step(epoch)
        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        #loss
        fake_B = generator(real_A)
        pred_fake_list = discriminator(fake_B)
        pred_real_list = discriminator(real_B)
        pred_fake = pred_fake_list[-1]
        
        #adv loss 
        true_labels =Variable( torch.Tensor(np.ones(pred_fake.size())).cuda(),  requires_grad=False)
        loss_GAN = criterion_GAN(pred_fake, true_labels)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        # vgg19 pertual loss
        loss_vgg  = criterion_Vgg(fake_B, real_B)
        #disc multi scale loss
        loss_disc_mutiscale = criterion_DiscMultiScaleLoss(pred_fake_list, pred_real_list)
        # Total loss
        loss_G = loss_GAN  + args.lambda_vgg*loss_vgg  + args.lambda_pixel * loss_pixel + loss_disc_mutiscale
                
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Real loss
        pred_real_list = discriminator(real_B)
        true_labels =Variable( torch.Tensor(np.ones(pred_real_list[-1].size() )).cuda(),  requires_grad=False)
        loss_real = criterion_GAN(pred_real_list[-1], true_labels)

        # Fake loss
        pred_fake_list = discriminator(fake_B.detach())
        fake_labels =Variable( torch.Tensor(np.zeros(pred_fake_list[-1].size() )).cuda(),  requires_grad=False)
        loss_fake = criterion_GAN(pred_fake_list[-1], fake_labels)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = args.epoch_num * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()


        # Print log
        sys.stdout.write("\r[Epoch%d/%d]-[Batch%d/%d]-[Dloss:%f]-[Gloss:%f, loss_pixel:%f, adv:%f, disc_multi:%f, vgg_pertual:%f] ETA:%s" %
                                                        (epoch+1, args.epoch_num,
                                                        i, len(train_dataloader),
                                                        loss_D.data.cpu(), loss_G.data.cpu(),
                                                        loss_pixel.data.cpu(), loss_GAN.data.cpu(), 
                                                        loss_disc_mutiscale.data.cpu(),
                                                        loss_vgg.data.cpu(),
                                                        time_left))
         #log in tensorboard                                               
        if count %20 ==0:
            writer.add_scalar('loss_G', float(loss_G.data.cpu()), count )
            writer.add_scalar('loss_D', float(loss_D.data.cpu()), count )

        # If at sample interval save image
        if batches_done % args.sample_interval == 0:
            sample_images(generator, test_dataloader, args, epoch, batches_done)


    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), './Exp1_RobotCarReal-deraindrop/%s/generator_%d.pth' % (args.model_result_dir, epoch) )
        torch.save(discriminator.state_dict(), './Exp1_RobotCarReal-deraindrop/%s/discriminator_%d.pth' % (args.model_result_dir, epoch))



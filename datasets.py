import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import torch
import h5py


IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_dataset(dir):
    images = []
    if not os.path.isdir(dir):
        raise Exception('Check dataroot')
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                item = path
                images.append(item)

    return images



class ImageDataset(Dataset):
    def __init__(self, args, gt_path, rain_path , transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.args = args
        #self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        self.gt_path = gt_path
        self.rain_path = rain_path
        self.gt_imgs_list = make_dataset(gt_path)
        self.rain_imgs_list = make_dataset(rain_path)
        self.mode =mode
    def __getitem__(self, index):

        split = self.rain_imgs_list[index].split('/')
        num = split[-1].split('_')[0]

        rain_img = Image.open(self.rain_imgs_list[index])
        if self.mode == 'train':
            gt_img = Image.open(self.gt_path+'/'+num+'_clean.png')
        else:
            gt_img = Image.open(self.gt_path+'/'+num+'_clean.jpg')  #test

        #print('rain_img:', self.rain_imgs_list[index])
        #print('gt_img:', self.gt_path+'/'+num+'_clean.png')


        if self.args.which_direction == 'AtoB':
            #img_A = img.crop((0, 0, w/2, h))
            #img_B = img.crop((w/2, 0, w, h))
            pass
        else:   #B2A  B rain 
            # img_B = img.crop((0, 0, w/2, h))   #rain
            # img_A = img.crop((w/2, 0, w, h))   #gt
            img_B = rain_img
            img_A = gt_img 

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.rain_imgs_list)



def read_h5_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        return data



#robotcar real dataset 
class RobotCar_Real_Dataset(Dataset):
    def __init__(self, label_path,  gt_data_path, inpt_path, rain_data_path, image_size):
        super(RobotCar_Real_Dataset, self).__init__()

        self.inpt_path = inpt_path
        self.label_path = label_path
        self.gt_name_list = read_h5_data(gt_data_path)
        self.rain_name_list = read_h5_data(rain_data_path)
        self.image_size = image_size
    def __len__(self):
        return len(self.rain_name_list)

    def __getitem__(self, index):
        randx = random.randint(0, 689 - self.image_size)
        randy = random.randint(0, 775 - self.image_size)
        rain_image = cv2.imread(self.inpt_path + self.rain_name_list[index].decode())[randx:randx + self.image_size, randy:randy + self.image_size, :]
        gt_image = cv2.imread(self.label_path + self.gt_name_list[index].decode())[randx:randx + self.image_size, randy:randy + self.image_size, :]
                
        rain_image = (rain_image / 255.0).astype('float32')
        gt_image = (gt_image / 255.0).astype('float32')
        rain_image = rain_image.transpose(2, 0, 1)
        gt_image = gt_image.transpose(2, 0, 1)
        return torch.Tensor(rain_image), torch.Tensor(gt_image)





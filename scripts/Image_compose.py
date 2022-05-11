#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from PIL import Image
from tqdm import tqdm



def image_compose(path, imagename):
    IMAGE_SIZE = 224  
    IMAGE_ROW = 2  
    IMAGE_COLUMN = 3  
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  
    ii = 0

    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(path[ii] + imagename).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
            ii = ii + 1
    return to_image  

if __name__ == '__main__':

    # path for projection images of 6 perpendicular planes
    t_imgpath = ['/data/datasets/SJTU-PCQA/projection/projection1/', 
                 '/data/datasets/SJTU-PCQA/projection/projection2/', 
                 '/data/datasets/SJTU-PCQA/projection/projection3/',
                 '/data/datasets/SJTU-PCQA/projection/projection4/', 
                 '/data/datasets/SJTU-PCQA/projection/projection5/', 
                 '/data/datasets/SJTU-PCQA/projection/projection6/']
                 
    # path for the dataset labels         
    t_txtpath = '/data/datasets/label-PCQA/label.txt'

    # saving path for the splicing images
    save_path = '/data/datasets/SJTU-PCQA/projection/projection_splicing/'

    t_fh = open(t_txtpath, 'r')
    t_imgs = []
    t_labels = []

    for line in t_fh:
        line = line.rstrip()
        words = line.split()
        t_imgs.append(words[0])


    Num_Image = len(t_imgs)
    for i in tqdm(range(Num_Image)):
        t_img1 = image_compose(t_imgpath, t_imgs[i])
        t_img1.save(save_path+t_imgs[i])





# -*- coding: utf-8 -*-
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize



def get_four_pictures():
    pictures = []
    for i in range(1, 5):
        img_path = 'prompt9_info/' + str(i) + '.png'
        img = imread(img_path)
        resize_img = imresize(img, [256, 256], interp='nearest')
        pictures.append(resize_img)
    return pictures


def get_picture():
    img = imread('prompt9_info/'+'守株待兔.jpg')
    #print img.shape
    resize_img = imresize(img, [256, 256], interp='nearest')
    print resize_img.shape
    return resize_img / 255.0





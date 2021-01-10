import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import os
import random
import imageio
import glob
from PIL import Image
import skimage.transform

def get_image_label(train_folder,filename):
    f = open(filename,'r').readlines()
    imgslst = []
    labels = []

    for row in f:
        sp = row.strip().split()
        imgslst.append(train_folder+sp[0]+'.jpg')
        if sp[1]=='0':
            labels.append([0,1])
        else:
            labels.append([1,0])

    return imgslst,labels


def get_img(imgslst,batch_idx,input_size, crop_window_size):
    img = imageio.imread(imgslst[batch_idx])
    h,w,c = img.shape
    hw = min(h,w)
    
    # crop image into square image
    if h>w:
        w_off = 0
        h_off = int((h-w)/2)+random.randint((-1)*crop_window_size,crop_window_size) 
        img = img[h_off:h_off+hw,w_off:w_off+hw,:]
    if h<w:
        h_off = 0
        w_off = int((w-h)/2)+random.randint((-1)*crop_window_size,crop_window_size) 
        img = img[h_off:h_off+hw,w_off:w_off+hw,:]
        
    img = skimage.transform.resize(img,[input_size,input_size,3])
    
    return img


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def get_gray_img(imgslst,batch_idx,input_size,crop_window_size):
    img = imageio.imread(imgslst[batch_idx])
    h,w,c = img.shape
    hw = min(h,w)

    # crop image into square image
    if h>w:
        w_off = 0
        h_off = int((h-w)/2)+random.randint((-1)*crop_window_size,crop_window_size)
        img = img[h_off:h_off+hw,w_off:w_off+hw,:]
    if h<w:
        h_off = 0
        w_off = int((w-h)/2)+random.randint((-1)*crop_window_size,crop_window_size)
        img = img[h_off:h_off+hw,w_off:w_off+hw,:]
        
    img = skimage.transform.resize(img,[input_size,input_size,3])
    img = rgb2gray(img)
    img = np.stack((img,)*3, axis=-1)
    
    return img


                
def load_checkpoint(sess,checkpoint_dir, iteration=None,model_name='NET.model'):
        print(" [*] Reading checkpoints...")
        print(type(checkpoint_dir) ,checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and iteration:
            # Restores dump of given iteration
            ckpt_name = model_name + '-' + str(iteration)
        elif ckpt and ckpt.model_checkpoint_path:
            # Restores most recent dump
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)

        ckpt_file = os.path.join(checkpoint_dir, ckpt_name)
        print('Reading variables to be restored from ' + ckpt_file)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, ckpt_file)
        return ckpt_name


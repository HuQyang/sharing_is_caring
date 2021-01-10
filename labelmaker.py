
import numpy as np
import os
import random
import imageio
import glob

import csv

def simplify_label():
    filename = 'ISIC/ISIC_2019_Training_GroundTruth.csv'
    f = open('ISIC/simplelabel.txt','w')

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=' ')
        line_count = 0
        for row in csv_reader:
            sp = row[0].split(',')
            img_name = sp[0]

            # we consider the catogory 1,3,8 as maglinant
            # we consider the catofory 2,4,5,7 as benign
            # catogory 6 and 9 are not clear as maglinant or benign, we drop them
            mal = [sp[i] for i in [1,3,8]] 
            ben = [sp[i] for i in [2,4,5,7]] 

            if any(x=='1.0' for x in mal):
                f.write(sp[0]+' '+'0'+'\n')
            if any(x=='1.0' for x in ben):
                f.write(sp[0]+' '+'1'+'\n')


def illumination_selected_imglst():
    dark_img_name = 'dark/basal1.jpg'
    dark_imgs = imageio.imread(dark_img_name)
    ref_intensity = np.mean(dark_imgs)

    train_folder = 'ISIC/ISIC_2019_Training_Input/' 
    filename = 'ISIC/simplelabel.txt'

    f_train = open('dark_train.txt','w')
    f_test = open('dark_test.txt','w')

    f = open(filename,'r').readlines()
    imgslst = []
    labels = [[0,0]]*len(f)
    count = 0
    for row in f:
        sp = row.strip().split()
        img_name = os.path.join(train_folder,sp[0]+'.jpg')
        img = imageio.imread(img_name)

        if np.mean(img)> ref_intensity:
            f_train.write(sp[0]+' '+sp[1]+'\n')
        else:
            f_test.write(sp[0]+' '+sp[1]+'\n')


def random_selected_imglst():
    filename = 'ISIC/simplelabel.txt'
    imglsts = open(filename,'r').readlines()
    f_save_train = open('random1_train.txt','w')
    f_save_test = open('random1_test.txt','w')

    f_train = open('dark_train.txt','r').readlines()
    random.shuffle(imglsts)

    lst_train = imglsts[0:len(f_train)]
    lst_test = imglsts[len(f_train):]

    f_save_train.write(''.join(lst_train))
    f_save_test.write(''.join(lst_test))

simplify_label()
illumination_selected_imglst()
random_selected_imglst()
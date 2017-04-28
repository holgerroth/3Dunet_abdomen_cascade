# -*- coding: utf-8 -*-
"""
Created on 2016

@author: rothhr
"""

## Input output
img_dir = '/media/CTimages'
img_search = '.nii.gz'
label_dir = None
label_search = ''
output_root = '/media/auto3DUNET_abdomen_TEST--3dUnet'

######################## FUNCTIONS ###############################
import os
import sys
from deploy_cascade import deploy_cascade
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import recursive_glob, recursive_glob2

images_with_no_labels = []
images = recursive_glob(img_dir,img_search)

DEVICE=0
for image in images:
    basename = os.path.splitext(os.path.basename(image))[0]
    basename = basename.replace('series','')
    if label_dir is not None:
        label = recursive_glob2(label_dir,basename,'.raw')
    else:
        label = []
    if len(label) != 1:
        print('[WARNING] No unique label found for {}'.format(image))
        images_with_no_labels.append(image)
        label = None
    else:
        label = label[0]
    print('image: {}'.format(image))
    print('label: {}'.format(label))
    deploy_cascade(image,label,output_root,DEVICE)
    
print('{} images_with_no_labels:'.format(len(images_with_no_labels)))    
for i in images_with_no_labels:
    print(i)

# -*- coding: utf-8 -*-
"""
Created on 2016

@author: rothhr
"""

##################################################################
######## Visceral #######################
########################################################
img_dir = '/home/rothhr/Data/Visceral/visceral-dataset/Anatomy3-trainingset/Volumes'
img_search = '_1_CTce_ThAb.nii.gz' # contrast enhanced CT
label_dir = None
output_root = '/home/rothhr/Data/Visceral/auto3DUNET_visceral'

######################## FUNCTIONS ###############################
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import recursive_glob, recursive_glob2
from deploy_cascade_visceral import deploy_cascade

images_with_no_labels = []
images = recursive_glob(img_dir,img_search)

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
    deploy_cascade(image,label,output_root)
    
print('{} images_with_no_labels:'.format(len(images_with_no_labels)))    
for i in images_with_no_labels:
    print(i)

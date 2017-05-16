# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:03:03 2016

@author: rothhr
"""

N_INSTANCES = 2

CURR_INSTANCE = 2


rBody = 2

##############################################################################################################
##############################################################################################################
# Visceral - fine-tune from stage2 ACC #######################################################
##############################################################################################################
#IGNORE_VALUE=255
#USE_BODY=True
#FLIP_DATA='Visceral'
#RESAMPLE_DATA = False# [0.6718, 0.6718, 0.501327]
#in_label_dir = '/home/rothhr/Data/Visceral/visceral-dataset/Anatomy3-trainingset/MergedSegmentations'
#in_image_dir = '/home/rothhr/Data/Visceral/visceral-dataset/Anatomy3-trainingset/Volumes'
#in_label_search = '_1_CTce_ThAb.uc.nii.gz'
#in_image_search = '_1_CTce_ThAb.nii.gz'
#IGNORE_GT = False
### Visceral (ACC online network, Stage 1) #######################################################
##ZERO_MEAN=False
##NORM=True
##LOAD_MASK = False; RESAMPLE_MASK = False
##DOWNSAMPLE=True
##DILATE_MASK_TO_INCLUDE = 0 # number iterations for dilation
##CROP = False
##SWAP_LABELS = None
##out_dir = '/media/rothhr/SSD/Visceral/Data_visceral-normed-down2'
##out_suffix = '_balanced-normed-down2.h5'
##out_list = '/media/rothhr/SSD/Visceral/tmp_' + out_suffix.replace('.h5','.list')
##win_min=1500 -2000
##win_max=2500 -2000
##dx = 2
##dy = 2
##dz = 2
#
## Visceral (ACC online network, Stage 2) #######################################################
#ZERO_MEAN=False
#NORM=True
#FLIP_DATA = 'Visceral'
#IGNORE_GT = False
#LOAD_MASK = False; RESAMPLE_MASK = False
#DOWNSAMPLE=True
#CROP = False
#SWAP_LABELS = None
#DILATE_MASK_TO_INCLUDE = 5 # number iterations for dilation
#LOAD_MASK = True; RESAMPLE_MASK = True
#in_mask_dir = '/media/rothhr/SSD/Visceral/3dUnet--Data_visceral-normed-down2/TRAIN/iter_130000'
##in_mask_dir = '/media/rothhr/SSD/Visceral/3dUnet--Data_visceral-normed-down2/VALID/iter_130000'
#in_mask_search = '--PRED.nii.gz' 
#out_dir = '/media/rothhr/SSD/Visceral/Data_Visceral-normed-down2-Stage2_dilGT'+str(DILATE_MASK_TO_INCLUDE)
#out_suffix = '_balanced-normed-down2-Stage2_dilGT'+str(DILATE_MASK_TO_INCLUDE)+'.h5'
#out_list = '/media/rothhr/SSD/Visceral/tmp_' + out_suffix.replace('.h5','.list')
#win_min=1500 -2000
#win_max=2500 -2000
#dx = 2
#dy = 2
#dz = 2

##############################################################################################################
############## TEST ON TORSO DATA #####################3
##############################################################################################################
IGNORE_VALUE=255
USE_BODY=True
FLIP_DATA=None
RESAMPLE_DATA = False# [0.6718, 0.6718, 0.501327]
in_label_dir = '/home/rothhr/Data/Torso/Data/Labels'
in_image_dir = '/home/rothhr/Data/Torso/Data/Images'
in_label_search = '.uc_raw.nii.gz'
in_image_search = '.nii.gz'
IGNORE_GT = False
## Visceral on Torso  (ACC online network, Stage 1) #######################################################
ZERO_MEAN=False
NORM=True
LOAD_MASK = False; RESAMPLE_MASK = False
DOWNSAMPLE=True
DILATE_MASK_TO_INCLUDE = 0 # number iterations for dilation
CROP = False
SWAP_LABELS = None
out_dir = '/media/rothhr/SSD/Torso/Data_visceral-normed-down2_Stage1'
out_suffix = '_balanced-normed-down2_Stage1.h5'
out_list = '/media/rothhr/SSD/Torso/tmp_' + out_suffix.replace('.h5','.list')
win_min=1500 #-2000
win_max=2500 #-2000
dx = 2
dy = 2
dz = 2

# Visceral on Torso (ACC online network, Stage 2) #######################################################
ZERO_MEAN=False
NORM=True
FLIP_DATA = None
IGNORE_GT = False
LOAD_MASK = False; RESAMPLE_MASK = False
DOWNSAMPLE=True
CROP = False
SWAP_LABELS = None
DILATE_MASK_TO_INCLUDE = 5 # number iterations for dilation
LOAD_MASK = True; RESAMPLE_MASK = True
in_mask_dir = '/media/rothhr/SSD/Torso/3dUnet--Data_visceral-normed-down2_Stage1/iter_130000'
in_mask_search = '--PRED.nii.gz' 
out_dir = '/media/rothhr/SSD/Torso/Data_visceral-normed-down2-Stage2_dilGT'+str(DILATE_MASK_TO_INCLUDE)
out_suffix = '_balanced-normed-down2-Stage2_dilGT'+str(DILATE_MASK_TO_INCLUDE)+'.h5'
out_list = '/media/rothhr/SSD/Torso/tmp_' + out_suffix.replace('.h5','.list')
win_min=1500 #-2000
win_max=2500 #-2000
dx = 2
dy = 2
dz = 2



##########
crop_marginx = 0
crop_marginy = 0
crop_marginz = 0
SAVE_NII_DATA = True



######### RUN ###########################
import nibabel as nib
import numpy as np
import h5py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import recursive_glob, recursive_glob2
from scipy import ndimage as ndi
import skimage.morphology
import skimage.measure

def convert_image_and_label_to_h5(image_file,label_file,out_file,mask_file=None):
    if not os.path.isfile(image_file):
        raise ValueError('image file does not exist: {}'.format(image_file))
    if not os.path.isfile(label_file):
        raise ValueError('label file does not exist: {}'.format(label_file))    

    print('image: {}\nlabel: {}\nout: {}\nmask: {}'.format(image_file,label_file,out_file,mask_file))
    outdir = os.path.split(out_file)[0]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    img = nib.load(image_file)
    print('image',img.shape,img.get_data_dtype())
    label = nib.load(label_file)
    print('label',label.shape,label.get_data_dtype())
    
    L = label.get_data()
    I = np.asarray(img.get_data(),dtype=np.float32)    

    if FLIP_DATA is 'NIH':
        L = L[::-1,::,::-1]
        I = I[::-1,::,::-1]
    elif FLIP_DATA is 'Visceral':
        L = L[::,::-1,::]
        I = I[::,::-1,::]
    else:
        print('No flipping')
        
    if RESAMPLE_DATA:
        size0 = I.shape
        spacing0 = np.abs(img.affine.diagonal()[0:3])
        sizeI = np.round(np.divide(np.multiply(size0,spacing0),RESAMPLE_DATA))
        xi = np.linspace(0,size0[0]-1,sizeI[0])
        yi = np.linspace(0,size0[1]-1,sizeI[1])
        zi = np.linspace(0,size0[2]-1,sizeI[2])
        XI, YI, ZI = np.meshgrid(xi, yi, zi)            
        print('Interp3 of IMAGE shape {} to Image shape {}'.format(size0,sizeI))
        I = interp3([0,size0[0]],[0,size0[1]],[0,size0[2]],I,\
                       XI, YI, ZI,\
                       method="linear") 
        print('Interp3 of LABEL shape {} to Image shape {}'.format(size0,sizeI))
        L = interp3([0,size0[0]],[0,size0[1]],[0,size0[2]],L,\
                       XI, YI, ZI,\
                       method="nearest") 
                       
    if np.any((np.asarray(img.get_shape())-np.asarray(label.get_shape()))!=0):
        raise ValueError('image and label have different dimensions!')
        
    #hx = int(label.shape[0]/2)
    #hy = int(label.shape[1]/2)
    #hz = int(label.shape[2]/2)
    if DOWNSAMPLE:
        L = L[::dx,::dy,::dz]
        I = I[::dx,::dy,::dz]
        print(' downsampled with ({},{},{}) to {}'.format(dx,dy,dz,np.shape(L)))        

    # only learn under body mask 
    if USE_BODY:
        print('USE BODY MASK')
        BODY = (I>=win_min)# & (I<=win_max)
        if np.sum(BODY)==0:
            raise ValueError('BODY could not be extracted!')
        # Find largest connected component in 3D
        struct = np.ones((3,3,3),dtype=np.bool)
        BODY = ndi.binary_erosion(BODY,structure=struct,iterations=rBody)
        BODY_labels = skimage.measure.label(np.asarray(BODY, dtype=np.int))
        props = skimage.measure.regionprops(BODY_labels)
        areas = []
        for prop in props: 
            areas.append(prop.area)
        # only keep largest, dilate again and fill holes                
        BODY = ndi.binary_dilation(BODY_labels==(np.argmax(areas)+1),structure=struct,iterations=rBody)
        # Fill holes slice-wise
        for z in range(0,BODY.shape[2]):    
            BODY[:,:,z] = ndi.binary_fill_holes(BODY[:,:,z])                    
    else:
        BODY = np.ones(I.shape,dtype=np.uint8) > 0
        print('USE ALL VOXELS...')

    if LOAD_MASK:
        print('load mask from {}'.format(mask_file))
        MASK = nib.load(mask_file).get_data()
        print(np.shape(MASK))        
        if RESAMPLE_MASK:
            xi = np.linspace(0,MASK.shape[0]-1,I.shape[0])
            yi = np.linspace(0,MASK.shape[1]-1,I.shape[1])
            zi = np.linspace(0,MASK.shape[2]-1,I.shape[2])
            XI, YI, ZI = np.meshgrid(xi, yi, zi)            
            print('Interp3 of MASK shape {} to Image shape {}'.format(MASK.shape,I.shape))
            MASK = interp3([0,MASK.shape[0]],[0,MASK.shape[1]],[0,MASK.shape[2]],MASK,\
                           XI, YI, ZI,\
                           method="nearest")        
        if np.any(MASK.shape!=I.shape):                           
            raise ValueError('Upsampling mask did not work! MASK shape {} to Image shape {}'.format(MASK.shape,I.shape))
        nib.save( nib.Nifti1Image(np.asarray(MASK,dtype=np.uint8),np.eye(4)), out_file.replace('.h5','--mask-interp.nii.gz') )                
    
        if not RESAMPLE_MASK and DOWNSAMPLE:
            MASK = MASK[::dx,::dy,::dz]
            print(' downsampled mask with ({},{},{}) to {}'.format(dx,dy,dz,np.shape(MASK)))              
        MASK = MASK>0  # use all foreground  
    else:             
        MASK = np.ones(I.shape,dtype=np.uint8) > 0
        print('USE NO MASK...')
        
    if DILATE_MASK_TO_INCLUDE>0:
        struct = np.ones((3,3,3),dtype=np.bool)
        print('Dilate MASK>0 with {} iterations...'.format(DILATE_MASK_TO_INCLUDE))
        MASK = ndi.binary_dilation(MASK>0,structure=struct,iterations=DILATE_MASK_TO_INCLUDE) > 0        

    MASK = MASK & BODY    
    MASK0 = np.copy(MASK) # This is saved as *--mask.nii.gz for later candidate generation
    MASK[L>0] = True # make sure labels are within mask!        
                
    # cropp based on largest connected component in mask    
    if CROP:
        MASK_labels = skimage.measure.label(np.asarray(MASK, dtype=np.int))
        props = skimage.measure.regionprops(MASK_labels)
        areas = []
        for prop in props: 
            areas.append(prop.area)
        # only keep largest          
        MASK = MASK_labels==(np.argmax(areas)+1)
        xyz = np.asarray(np.where(MASK>0),dtype=np.int)
        print('Cropping based on indices {}'.format(np.shape(xyz)))
        minx = np.min(xyz[0,::])
        maxx = np.max(xyz[0,::])
        miny = np.min(xyz[1,::])
        maxy = np.max(xyz[1,::])
        minz = np.min(xyz[2,::])
        maxz = np.max(xyz[2,::])        
        print('  found ranges x: {} to {}'.format(minx,maxx))
        print('               y: {} to {}'.format(miny,maxy))
        print('               z: {} to {}'.format(minz,maxz))
        L = L[minx:maxx+1,miny:maxy+1,minz:maxz+1]
        I = I[minx:maxx+1,miny:maxy+1,minz:maxz+1]
        MASK = MASK[minx:maxx+1,miny:maxy+1,minz:maxz+1]
        MASK0 = MASK0[minx:maxx+1,miny:maxy+1,minz:maxz+1]
        print(' cropped to {}'.format(np.shape(L)))        
        with open(out_file.replace('.h5','--crop.txt'), 'w') as f:
            f.write('dim, min, max\n')
            f.write('x, {}, {}\n'.format(minx,maxx))
            f.write('y, {}, {}\n'.format(miny,maxy))
            f.write('z, {}, {}\n'.format(minz,maxz))        
        
    Nvalid = np.sum(MASK)
    Nvoxels = np.size(MASK)
    print('Use {} of {} voxels within mask ({} %)'.format(Nvalid,Nvoxels,100*float(Nvalid)/Nvoxels))
    assert(Nvalid>0)
        
    # correct label image    
    L = np.asarray(L, np.uint8) # use anything larger 0
    if SWAP_LABELS is not None:
        if len(SWAP_LABELS) != 2:
            raise ValueError('SWAP_LABELS only supports 2 labels!')
        xyz0 = np.asarray(np.nonzero(L==SWAP_LABELS[0])).T
        xyz1 = np.asarray(np.nonzero(L==SWAP_LABELS[1])).T
        if np.ptp(xyz1) > np.ptp(xyz0): # assume atery should larger extent (in all directions)
            Ltmp = np.copy(L)
            f = open(out_file.replace('.h5','--swapped.log'), 'w')
            f.close()
            print('swap {}...'.format(SWAP_LABELS))
            L[Ltmp==SWAP_LABELS[0]] = SWAP_LABELS[1]    
            L[Ltmp==SWAP_LABELS[1]] = SWAP_LABELS[0]    
        else:
            print('do not swap labels...')
    
    
    L[~MASK] = IGNORE_VALUE
    l, lc = np.unique(L,return_counts=True)
    lc = lc[l!=IGNORE_VALUE] 
    l = l[l!=IGNORE_VALUE]
    print('Labels')
    frac = []
    for cidx, c in enumerate(lc):
        print(cidx)
        frac.append(float(c)/Nvalid)        

    # compute weights that sum up to 1        
    # generate balanced weight
    weights = np.ndarray(np.shape(I),dtype=np.float32)
    weights.fill(0.0)
    w = []
    for cidx, c in enumerate(lc):        
        wc = (1.0-frac[cidx])/(len(lc)-1) # 
        w.append(wc)
        print('  {}: {} of {} ({} percent, w={})'.format(l[cidx],c,np.size(L),100*float(c)/np.size(L),wc))
        weights[L==l[cidx]] = wc
    print('sum(w) = {}'.format(np.sum(w)))        
    if np.abs(1.0-np.sum(w)) > 1e-8:
        print('sum(w) != 1.0, but {}'.format(np.sum(w)))
    weights[~MASK] = 0.0 # ignore in cost function but also via label IGNORE_VALUE
    
    # image windowing
    print('min/max data: {}/{}'.format(np.min(I),np.max(I)))
    I[I<win_min] = win_min
    I[I>win_max] = win_max
    I = I-np.min(I) 
    I = I/np.max(I)
    print('min/max windowed: {}/{}, mean {}'.format(np.min(I),np.max(I),np.mean(I)))
    
    if NORM:
        # assume I is already scaled between 0 to 1
        I = 2.0*I-1.0
        print('min/max normed: {}/{}, mean {}'.format(np.min(I),np.max(I),np.mean(I)))
    
    if ZERO_MEAN:
        I = I - np.mean(I[MASK])
        print('ZERO MEAN: {},min/max normed: {}/{}'.format(np.mean(I),np.min(I),np.max(I)))
    
    if np.any(np.asarray(np.shape(I))-np.asarray(np.shape(L))):
        raise ValueError('image and label have different sizes!')
    

    print('min/max weights: {}/{}'.format(np.min(weights),np.max(weights)))
    
    print('save nifti images.')  
    if SAVE_NII_DATA:
        nib.save( nib.Nifti1Image(I,np.eye(4)), out_file.replace('.h5','--data.nii.gz') )
    nib.save( nib.Nifti1Image(L,np.eye(4)), out_file.replace('.h5','--label.nii.gz') )
    nib.save( nib.Nifti1Image(weights,np.eye(4)), out_file.replace('.h5','--weights.nii.gz') )
    nib.save( nib.Nifti1Image(np.asarray(MASK0,dtype=np.uint8),np.eye(4)), out_file.replace('.h5','--mask.nii.gz') )    
    
    print('save h5 as {}...'.format(out_file))
    with h5py.File(out_file,'w') as h5f:
        #h5f.create_dataset('data',data=img.get_data()[np.newaxis,np.newaxis,0:nx,0:ny,0:nz],dtype=np.short) # int16
        #h5f.create_dataset('label',data=L[np.newaxis,np.newaxis,0:nx,0:ny,0:nz],dtype=np.uint8)
        #h5f.create_dataset('weights',data=weights[np.newaxis,np.newaxis,0:nx,0:ny,0:nz],dtype=np.float16)    
    
        # caffe input: n * c_i * h_i * w_i *d_i
        I = np.transpose(I,(2,1,0)) # caffe format    
        L = np.transpose(L,(2,1,0)) # caffe format
        weights = np.transpose(weights,(2,1,0)) # caffe format
    
        h5f.create_dataset('data',data=I[np.newaxis,np.newaxis,:,:,:],dtype=np.float32) # int16
        h5f.create_dataset('label',data=L[np.newaxis,np.newaxis,:,:,:],dtype=np.uint8)
        h5f.create_dataset('weights',data=weights[np.newaxis,np.newaxis,:,:,:],dtype=np.float32)    
        
        print('saved data ',np.shape(h5f.get('data')))
        print('saved label ',np.shape(h5f.get('label')))
        print('saved weights ',np.shape(h5f.get('weights')))
        
        #h5f.create_dataset('data',data=img.get_data(),dtype='f4') # int16
        #h5f.create_dataset('label',data=L,dtype='f4')    
        #h5f.create_dataset('weights',data=weights,dtype='f4')    
             
    print('...done.')

### MAIN FUNCTION ##########                
def main(n_instances, curr_instance):
    label_files = recursive_glob(in_label_dir,in_label_search)    

    N = len(label_files)    
    if N>0:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)        
        #flist = open(out_list, 'w', 0)
        flist = open(out_list, 'a', 0) #append to existing
    else:
        raise ValueError('No files found!')
    
    Nstep = np.ceil(float(N)/n_instances);
    for_start = int( (curr_instance-1)*Nstep )
    for_end = int( for_start + Nstep)
    if for_end>N:
        for_end = N
    print('for_start:for_end is {}:{}'.format(for_start,for_end))
    
    converted = 0
    already_computed = 0
    img_pair_not_found = 0
    skip_invalid = 0
    failed_count = 0
    for i, label_file in enumerate(label_files[for_start:for_end]):
        print('{} of {}: {}'.format(i+1,N,label_file))            
        basename = os.path.split(label_file.replace(in_label_search,''))[1]
        basename = basename[0:basename.find('_')].lower()
        img_file = recursive_glob2(in_image_dir,basename,in_image_search)[0]
        curr_dir = img_file.replace(in_image_dir,out_dir).replace(in_image_search,'')
        FOUND = False
        mask_file = None
        if LOAD_MASK:
            mask_file = recursive_glob2(in_mask_dir,in_mask_search,basename)
            if len(mask_file)!=1:
                FOUND = False    
                img_pair_not_found += 1
                print('  WARNING: no mask found for {}\n in {} with {} search'.format(label_file,in_mask_dir,basename))                    
                mask_file = None
                continue
    
        curr_dir = curr_dir.replace('.nii.gz','')
        curr_dir = curr_dir.replace('.nii','')
        out_file = curr_dir + '/' + os.path.split(curr_dir)[1] + out_suffix
        if not os.path.isfile(out_file):
            print(100*'=')
            try:
                if LOAD_MASK:
                    convert_image_and_label_to_h5(img_file,label_file,out_file,mask_file[0])        
                else:
                    convert_image_and_label_to_h5(img_file,label_file,out_file)
                    converted += 1
            except Exception as inst:
                print type(inst)     # the exception instance
                print inst.args      # arguments stored in .args
                print inst           # __str__ allows args to be printed directly                            
                failed_count += 1
                sys.exit()
        else:
            already_computed += 1
            print('  Already computed ({}): {}'.format(already_computed,out_file))                            
            flist.write('{}\n'.format(out_file)) 
        FOUND = True     
        if not FOUND:
            img_pair_not_found += 1
            print('  WARNING: no image found for {}'.format(label_file))
            
    print('Converted {} of {} file pairs.'.format(converted,N))
    print('Already converted {} of {} file pairs.'.format(already_computed,N))
    print('No suitable image found {} of {} file pairs.'.format(img_pair_not_found,N))
    print('Skip/invalid {} of {} file pairs.'.format(skip_invalid,N))
    print('Convert failed {} of {} file pairs.'.format(failed_count,N))
    flist.close()
     
def interp3(xrange, yrange, zrange, v, xi, yi, zi, **kwargs):
    #http://stackoverflow.com/questions/21836067/interpolate-3d-volume-with-numpy-and-or-scipy
    #from numpy import array
    from scipy.interpolate import RegularGridInterpolator as rgi

    x = np.arange(xrange[0],xrange[1])
    y = np.arange(yrange[0],yrange[1])
    z = np.arange(zrange[0],zrange[1])
    interpolator = rgi((x,y,z), v, **kwargs)
    
    pts = np.array([np.reshape(xi,(-1)), np.reshape(yi,(-1)), np.reshape(zi,(-1))]).T    
    Vi = interpolator(pts)
    return np.reshape(Vi, np.shape(xi)).transpose((1,0,2)) # skipy flips x,y     
        
if __name__ == '__main__':
    main(N_INSTANCES, CURR_INSTANCE)    





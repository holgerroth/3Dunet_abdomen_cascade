# -*- coding: utf-8 -*-
"""
Created on 2016

@author: rothhr
"""

in_image_search = '.nii.gz'
in_label_search = '.nii.gz'
win_min=-5000
win_max=5000 
img_dir = '/media/CTimages'
label_dir = '/media/CTimages_labels'
output_root = '/media/CTimages/CTimages_down2'
FLIP_DATA=None
USE_LABELS = [7]

##########################################################
ZERO_MEAN=False
NORM=True
IGNORE_VALUE=255

###### Common Stage 1
DOWNSAMPLE=True

USE_BODY=False
RESAMPLE_DATA = False# [0.6718, 0.6718, 0.501327]
IGNORE_GT = False
## Visceral on Torso  (ACC online network, Stage 1) #######################################################
ZERO_MEAN=False
NORM=True
DILATE_MASK_TO_INCLUDE = 0 # 
RESAMPLE_MASK = False
CROP = False
SWAP_LABELS = None
dx = 2
dy = 2
dz = 2
EXTRACT_FEATURES = False

##########
crop_marginx = 0
crop_marginy = 0
crop_marginz = 0
SAVE_NII_DATA = True

rBody = 2

N_INSTANCES = 1
CURR_INSTANCE = 1

######################## FUNCTIONS ###############################
import numpy as np
import os
import nibabel as nib
import h5py
#import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#from data import recursive_glob, recursive_glob2
import mori 
from scipy import ndimage as ndi
import skimage.morphology
import skimage.measure
from data import recursive_glob, recursive_glob2

def read_image_info(filename):
    basename = os.path.basename(filename)
    if '.nii' in basename:
        img = nib.load(filename)
        size = img.shape
        spacing = img.affine.diagonal()[0:3]
    elif '.header' in basename:
        hdr = mori.read_mori_header(filename)
        size = hdr['size']
        spacing = hdr['spacing']
    else:
        raise TypeError('Only nifti and mori header files supported! Not {}'.format(filename))
    return size,spacing

def read_image(filename,dtype=None):
    basename = os.path.basename(filename)
    if '.nii' in basename:
        img = nib.load(filename)
        spacing = img.affine.diagonal()[0:3]
        print('nifti:',img.shape,img.get_data_dtype(),filename)
        I = img.get_data()
    else: 
        I, hdr = mori.read_mori(filename,dtype)
        if hdr is not None:
            spacing = hdr['spacing']
        else:
            spacing = [1, 1, 1]
    print('{}: {}, spacing {}'.format(basename,np.shape(I),spacing))
    return I, spacing

############ Functions ###################
def convert_image_and_label_to_h5(image_file,label_file,out_file,\
                                  mask_file=None,DILATE_MASK_TO_INCLUDE=0,\
                                  win_min=0,win_max=5000,ZERO_MEAN=False,NORM=True):
    if not os.path.isfile(image_file):
        raise ValueError('image file does not exist: {}'.format(image_file))
    if label_file is not None and not os.path.isfile(label_file):
        raise ValueError('label file does not exist: {}'.format(label_file))    

    print('image: {}\nlabel: {}\nout: {}\nmask: {}'.format(image_file,label_file,out_file,mask_file))
    outdir = os.path.split(out_file)[0]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

            
    I, i_spacing = read_image(image_file,dtype='>u2') # dtype is only used with raw     
    I = np.asarray(I,dtype=np.float32)

    if label_file is not None:    
        L, l_spacing = read_image(label_file,dtype='>u1')
    else:
        L = np.zeros(np.shape(I),dtype=np.uint8)
        l_spacing = i_spacing
    
    if FLIP_DATA is 'NIH':
        L = L[::-1,::,::-1]
        I = I[::-1,::,::-1]
    elif FLIP_DATA is 'Visceral':
        L = L[::,::-1,::]
        I = I[::,::-1,::]
    elif FLIP_DATA is 'CTCompAnaB':
        I = I[::-1,::-1,::]        
    else:
        print('No flipping')
        
    if RESAMPLE_DATA:
        size0 = I.shape
        spacing0 = np.abs(i_spacing)
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
                       
    if np.any((np.asarray(I.shape)-np.asarray(L.shape))!=0):
        #raise ValueError('image and label have different dimensions!')
        print('[WARNING] image and label have different dimensions! => skip case')
        return
        
    #hx = int(label.shape[0]/2)
    #hy = int(label.shape[1]/2)
    #hz = int(label.shape[2]/2)
    if DOWNSAMPLE:
        L = L[::dx,::dy,::dz]
        I = I[::dx,::dy,::dz]
        print(' downsampled with ({},{},{}) to {}'.format(dx,dy,dz,np.shape(L)))        
        
    if USE_LABELS is not None:
        Ltmp = np.copy(L)
        L[...] = 0
        for use_idx,use_label in enumerate(USE_LABELS):
            print('USE_LABEL: map {} to {}'.format(use_label,use_idx+1))
            L[Ltmp==use_label] = use_idx+1

    # only learn under body mask 
    if USE_BODY:
        print('USE BODY MASK')
        BODY = (I>=win_min)# & (I<=win_max)
        print(' {} of {} voxels masked.'.format(np.sum(BODY),np.size(BODY)))
        if np.sum(BODY)==0:
            raise ValueError('BODY could not be extracted!')
        # Find largest connected component in 3D
        struct = np.ones((3,3,3),dtype=np.bool)
        BODY = ndi.binary_erosion(BODY,structure=struct,iterations=rBody)
        if np.sum(BODY)==0:
            raise ValueError('BODY mask disappeared after erosion!')        
        BODY_labels = skimage.measure.label(np.asarray(BODY, dtype=np.int))
        props = skimage.measure.regionprops(BODY_labels)
        areas = []
        for prop in props: 
            areas.append(prop.area)
        print('  -> {} areas found.'.format(len(areas)))
        # only keep largest, dilate again and fill holes                
        BODY = ndi.binary_dilation(BODY_labels==(np.argmax(areas)+1),structure=struct,iterations=rBody)
        # Fill holes slice-wise
        for z in range(0,BODY.shape[2]):    
            BODY[:,:,z] = ndi.binary_fill_holes(BODY[:,:,z])                    
    else:
        BODY = np.ones(I.shape,dtype=np.uint8) > 0
        print('USE ALL VOXELS...')

    if mask_file is not None:
        print('load mask from {}'.format(mask_file))
        MASK, m_spacing = read_image(mask_file,dtype='>u1')
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
    
        #if RESAMPLE_MASK and DOWNSAMPLE:
        if DOWNSAMPLE:            
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
    if len(lc)>1:
        for cidx, c in enumerate(lc):        
            wc = (1.0-frac[cidx])/(len(lc)-1) # 
            w.append(wc)
            print('  {}: {} of {} ({} percent, w={})'.format(l[cidx],c,np.size(L),100*float(c)/np.size(L),wc))
            weights[L==l[cidx]] = wc
    else:
        print('[WARNING] all voxels have the same label: {}'.format(lc))
        w.append(1.0)
        weights[...] = 1.0
    print('sum(w) = {}'.format(np.sum(w)))        
    if np.abs(1.0-np.sum(w)) > 1e-8:
        print('sum(w) != 1.0, but {}'.format(np.sum(w)))
    weights[~MASK] = 0.0 # ignore in cost function but also via label IGNORE_VALUE
    
    # image windowing
    print('min/max data: {}/{} => {}/{}'.format(np.min(I),np.max(I),win_min,win_max))
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
    return np.reshape(Vi, np.shape(xi))        
          
def prepare(img_file,label_file=None,output_root='/tmp',device=0):
    basename = os.path.splitext(os.path.basename(img_file))[0]
    output_root = os.path.join(output_root,basename)
    output_root1 = os.path.join(output_root,basename+'_stage1')   
    h5file1 = os.path.join(output_root1,basename+'_data1.h5')   
    
    if not os.path.isdir(output_root1):
        os.makedirs(output_root1)
        
    orig_size, orig_spacing = read_image_info(img_file)
        
    convert_image_and_label_to_h5(img_file,label_file,h5file1,None,0,win_min,win_max,ZERO_MEAN,NORM)    
  
def prepare2(img_file,label_file=None,output_root='/tmp',device=0):
    basename = os.path.splitext(os.path.basename(img_file))[0]
    output_root = os.path.join(output_root,basename)
    output_root1 = os.path.join(output_root,basename+'_stage2')   
    h5file1 = os.path.join(output_root1,basename+'_data2.h5')   
    
    if not os.path.isdir(output_root1):
        os.makedirs(output_root1)
        
    orig_size, orig_spacing = read_image_info(img_file)
        
    convert_image_and_label_to_h5(img_file,label_file,h5file1,label_file,DILATE_MASK_TO_INCLUDE,win_min,win_max,ZERO_MEAN,NORM)         
  
def main(n_instances, curr_instance):
    images_with_no_labels = []
    skip_count = 0;
    images = recursive_glob(img_dir,in_image_search)

    N = len(images)    
    Nstep = np.ceil(float(N)/n_instances);
    for_start = int( (curr_instance-1)*Nstep )
    for_end = int( for_start + Nstep)
    if for_end>N:
        for_end = N
    print('for_start:for_end is {}:{}'.format(for_start,for_end))    
    
    for idx in range(for_start,for_end):
        image = images[idx]
        if ('.d' in image) or ('.H' in image):
            print('SKIP ',image)
            skip_count += 1
            continue        
        basename = os.path.splitext(os.path.basename(image.replace(in_image_search,'')))[0]
        basename = basename.replace('PANCREAS_','')
        #label = recursive_glob2(label_dir,basename,in_label_search)
        label = [os.path.join(label_dir,basename+in_label_search)]
        if len(label) != 1:
            raise ValueError('No unique label found for {} ({})'.format(image,len(label)))
            print('[WARNING] No unique label found for {} ({})'.format(image,len(label)))
            images_with_no_labels.append(image)
            label = None
        else:
            label = label[0]
        print('image: {}'.format(image))
        print('label: {}'.format(label))
        prepare(image,label,output_root)
        ####prepare2(image,label,output_root)
        
    print('{} images_with_no_labels:'.format(len(images_with_no_labels)))    
    for i in images_with_no_labels:
        print(i)
    print('Skipped {} of {} images'.format(skip_count,N))
        
if __name__ == '__main__':
    main(N_INSTANCES, CURR_INSTANCE) 
    


    

# -*- coding: utf-8 -*-
"""
Created on 2016

@author: rothhr
"""

#################################################################33
######## Torso #######################
########################################################
#img_file = '/home/rothhr/Data/Torso/Data/Images/t0001468_series4.nii.gz'
#label_file = '/home/rothhr/Data/Torso/Data/Labels/T0001468_4.uc_raw.nii.gz'
#output_root = '/media/rothhr/SSD/Torso/TMP/3dUnet-full_pipline'

#img_file = '/media/rothhr/SSD/CTCompAna/RAW/t0000085_series7.header'
#label_file = '/media/rothhr/SSD/CTCompAna/RAW/t0000085_7_label_all.raw'
#output_root = '/media/rothhr/SSD/Torso/TMP/3dUnet-full_pipline_raw_hdr'

####################### COMMON ###################################
############## Stage 1: down 2: normed ###########################
proto_text1 = 'models/3dUnet_Abdomen_with_BN-test.prototxt'
trained_model1 = 'snapshot-Stage1/3dUnet_Abdomen_with_BN_zeromean_iter_200000.caffemodel'
############## Stage 1: down 2: normed ###########################
proto_text2 = 'models/3dUnet_Abdomen_with_BN-test.prototxt'#'models/3dUnet_Abdomen_with_BN-Stage2-test.prototxt'
trained_model2 = 'snapshot-Stage2/3dUnet_Abdomen_with_BN_normed-Stage2_iter_115000.caffemodel'

# Stage1
win_min1=1500
win_max1=2500
ZERO_MEAN1=True
NORM1=False
# Stage2
win_min2=0
win_max2=5000 # basically full range
ZERO_MEAN2=False
NORM2=True

##############################################################################################################
############## TEST ON ABDOMEN ACC DATA #####################3
##############################################################################################################
IGNORE_VALUE=255
USE_BODY=True
FLIP_DATA=None
RESAMPLE_DATA = False# [0.6718, 0.6718, 0.501327]
in_label_search = '.uc_raw.nii.gz'
in_image_search = '.nii.gz'
IGNORE_GT = False
## Visceral on Torso  (ACC online network, Stage 1) #######################################################
ZERO_MEAN=False
NORM=True
DILATE_MASK_TO_INCLUDE = 5 # number iterations for dilation in Stage2
RESAMPLE_MASK = False
DOWNSAMPLE=True
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

N_INSTANCES = 1
CURR_INSTANCE = 1
rBody = 2

######################## FUNCTIONS ###############################
import sys
sys.path.insert(0,'../caffe_unet_3D_v1.0_patch/caffe/python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import os
import nibabel as nib
from tqdm import tqdm
import h5py
#import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#from data import recursive_glob, recursive_glob2
import mori 
from scipy import ndimage as ndi
import skimage.morphology
import skimage.measure
import time

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
        print('nifti:',img.shape,img.get_data_dtype())
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
        MASK, m_spacing = read_image(mask_file)
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
    
        if RESAMPLE_MASK and DOWNSAMPLE:
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

def deploy(proto_text,trained_model,image,mask,outprefix,device=0):        
    ############ RUN ###################
    if not os.path.isfile(proto_text):
        raise ValueError('{} does not exist!'.format(proto_text))
    if not os.path.isfile(trained_model):
        raise ValueError('{} does not exist!'.format(proto_text))            
    if not os.path.isfile(image):
        raise ValueError('{} does not exist!'.format(proto_text))
    if not os.path.isfile(mask):
        raise ValueError('{} does not exist!'.format(proto_text))
            
    if os.path.isabs(trained_model):
        model_root = os.path.dirname(trained_model)
        trained_model = os.path.basename(trained_model)
    else:
        model_root = '.'
    
    ### MAIN ####
    #mean=(104.00699, 116.66877, 122.67892)
        
    #remove the following two lines if testing with cpu
    caffe.set_mode_gpu()
    caffe.set_device(device) 
    
    # load net
    print("load net from {} ...".format(trained_model))
    net = caffe.Net(proto_text, model_root+'/'+trained_model, caffe.TEST)
    
    print('net.inputs[0] = ',net.inputs[0])
    print('net.inputs[1] = ',net.inputs[1])
    print('net.outputs[0] = ',net.outputs[0])
    
    print('load image: {}'.format(image)) 
    img = nib.load(image)
    I = img.get_data()
    I = np.asarray(img.get_data(),dtype=np.float32)
    
    m = nib.load(mask)
    MASK = np.asarray(m.get_data(),dtype=np.bool)
    MASKin = np.transpose(MASK,(2,1,0))
    
    # Loop through image tiles:
    dim_data = np.asarray([I.shape[2],I.shape[1],I.shape[0]],dtype=np.float)
    input = net.blobs['def'].data[0]
    dim_input = np.asarray(np.shape(input)[0:-1],dtype=np.float)
    output = net.blobs['score'].data[0]
    NUM_CLASSES = np.shape(output)[0]
    print('NUM_CLASSES: {}'.format(NUM_CLASSES))
    dim_output = np.asarray(np.shape(output)[1::],dtype=np.float) # ignore class-dimension    
    #dim_bottom = np.asarray([7, 9, 9],dtype=np.int)
    dim_tiles = np.ceil(np.divide(dim_data,dim_output))
    dim_min_offset = -1*np.asarray([(dim_input[0]-dim_output[0])/2, \
                                    (dim_input[1]-dim_output[1])/2, \
                                    (dim_input[2]-dim_output[2])/2])
    dim_max_offset = dim_data + dim_min_offset - dim_output                                
    dim_ratio = np.divide(dim_input,dim_output)
    print('dimensions:\n data {}\n input {}\n output {}\n tiles {}\n min_offset {}\n dim_max_offset {}\n dim_ratio{}' \
        .format(dim_data,dim_input,dim_output,dim_tiles,dim_min_offset,dim_max_offset,dim_ratio))
        
    # set input to network   
    Iin = np.transpose(I,(2,1,0))
    net.blobs['data'].reshape(1, 1, *Iin.shape)
    net.blobs['data'].data[...] = Iin[np.newaxis,np.newaxis,:,:,:]    
        
    outdir = os.path.split(outprefix)[0]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)    
        
    dim_pred = np.asarray([int(dim_tiles[0]*dim_output[0]),\
                           int(dim_tiles[1]*dim_output[1]),\
                           int(dim_tiles[2]*dim_output[2])])
    print('dim_pred: {}'.format(dim_pred))
    Ntiles = np.prod(dim_tiles)
    print('Ntiles = ', Ntiles)
    
    # allocate outprefix
    print('Loop through {} (= {}) tiles in {} image...'.format(dim_tiles,int(Ntiles),dim_data))    
    PROBS = np.zeros((NUM_CLASSES,dim_pred[0],dim_pred[1],dim_pred[2]),dtype=np.float32)
    PROBS.fill(-1.0)
    PRED = np.zeros(dim_pred,dtype=np.short)
    PRED.fill(-1)
    TILES = np.zeros(dim_pred,dtype=np.short)
    if EXTRACT_FEATURES:
        FEATS = np.zeros((N_FEATURES,dim_pred[0],dim_pred[1],dim_pred[2]),dtype=np.float32)    
    #NETIMG = np.zeros((np.shape(I)[2],np.shape(I)[1],np.shape(I)[0]),dtype=np.float32)
    Def = np.mgrid[0:dim_input[0],0:dim_input[1],0:dim_input[2]]
    patch_nr = 0
    with tqdm(total=int(Ntiles)) as pbar:
        for i in range(0,int(dim_tiles[0])):       # width
          for j in range(0,int(dim_tiles[1])):     # height
            for k in range(0,int(dim_tiles[2])):   # level      
                x = np.arange(0,dim_input[0]) + dim_min_offset[0] + i*dim_output[0]
                y = np.arange(0,dim_input[1]) + dim_min_offset[1] + j*dim_output[1]
                z = np.arange(0,dim_input[2]) + dim_min_offset[2] + k*dim_output[2]
                x1 = int(np.min(x))
                x2 = int(np.max(x)+1)
                y1 = int(np.min(y))
                y2 = int(np.max(y)+1)
                z1 = int(np.min(z))
                z2 = int(np.max(z)+1)
                patch_nr += 1
                #print(80*'#')
                #print('Patch {} of {}: ({},{},{}) ==> ({},{},{})'.format(patch_nr,Ntiles,x1,y1,z1,x2,y2,z2))
                #print(' dimension: [{},{},{}]'.format(len(x),len(y),len(z)))        
                #print(80*'#')                       
                pbar.update(1)            
                
                   
                #print(x1,x2,y1,y2,z1,z2)
                #Patch = Iin[x1:x2,y1:y2,z1:z2]
        
                # identity transform        
                Def = np.mgrid[x1:x2,y1:y2,z1:z2]
                #print('Def = {}'.format(np.shape(Def)))    
                Def = np.transpose(Def, (1, 2, 3, 0))
                #print('Def = {}'.format(np.shape(Def)))   
                Def = Def[np.newaxis,:,:,:,:] 
                net.blobs['def'].data[...] = Def
                
                #print('net.blobs[''data''].data = {}'.format(net.blobs['data'].data.shape))
                #print('net.blobs[''def''].data = {}'.format(net.blobs['def'].data.shape))
                
                net.forward()
                out = net.blobs['score'].data[0]
                #nib.save( nib.Nifti1Image(np.transpose(out,(1,2,3,0)),np.eye(4)), outprefix+'--out.nii.gz')
                out = np.asarray(out,dtype=np.double) # convert to double
        
                #print('min/max response: {}/{}'.format(np.min(out),np.max(out)))
                #print(out.shape)
                #print('softmax:')
                out = softmax(out)
                #print('min/max softmax: {}/{}'.format(np.min(out),np.max(out)))
                #print(out.shape)            
                pred = np.argmax(out,axis=0)
                #pred = out[2,:,:,:]
                #print(out.shape)
                #print('min/max pred: {}/{}'.format(np.min(pred),np.max(pred)))
                                
                # DEBUG
                #nib.save( nib.Nifti1Image(np.asarray(pred,dtype=np.uint8),np.eye(4)), outprefix+'--pred.nii.gz')
                #print('saved DEBUG outprefix at {}'.format(outprefix))            
                
                xout = np.arange(0,dim_output[0]) + i*dim_output[0]
                yout = np.arange(0,dim_output[1]) + j*dim_output[1]
                zout = np.arange(0,dim_output[2]) + k*dim_output[2]
                xout1 = int(np.min(xout))
                xout2 = int(np.max(xout)+1)
                yout1 = int(np.min(yout))
                yout2 = int(np.max(yout)+1)
                zout1 = int(np.min(zout))
                zout2 = int(np.max(zout)+1)            
        
                #print('pred = {}'.format(np.shape(pred))) 
                PROBS[:,xout1:xout2,yout1:yout2,zout1:zout2] = out
                PRED[xout1:xout2,yout1:yout2,zout1:zout2] = pred
                
                # skip if all mask in output space are zero
                if np.sum(MASKin[xout1:xout2,yout1:yout2,zout1:zout2])==0.0:
                    continue             
                
                # make tile image
                TILES[xout1,yout1:yout2,zout1:zout2] = 1
                TILES[xout2-1,yout1:yout2,zout1:zout2] = 1
                TILES[xout1:xout2,yout1,zout1:zout2] = 1
                TILES[xout1:xout2,yout2-1,zout1:zout2] = 1
                TILES[xout1:xout2,yout1:yout2,zout1] = 1
                TILES[xout1:xout2,yout1:yout2,zout2-1] = 1
                
                if EXTRACT_FEATURES:
                    feats = net.blobs[feat_layer].data[0]               
                    #########feats = softmax(feats)
                    FEATS[:,xout1:xout2,yout1:yout2,zout1:zout2] = feats                
            #end for i in range(0,int(dim_tiles[0])):       # width
          #end for j in range(0,int(dim_tiles[1])):     # height
        #end for k in range(0,int(dim_tiles[2])):   # level                  
    #end with tqdm as pbar:                
    pbar.close()            
    #print(80*'#')
    #print(80*'#')    
    
    #nib.save( nib.Nifti1Image(np.transpose(PRED,(2,1,0)),np.eye(4)) , outprefix+'--pred.nii.gz')    
    #nib.save( nib.Nifti1Image(np.transpose(TILES,(2,1,0)),np.eye(4)) , outprefix+'--tiles.nii.gz')    
    #nib.save( nib.Nifti1Image(NETIMG,np.eye(4)), outprefix+'--NETIMG.nii.gz')  
    #nib.save( nib.Nifti1Image(np.transpose(PROBS,(1,2,3,0)),np.eye(4)), outprefix+'--probs.nii.gz')  
    print('PRED min/max response: {}/{}'.format(np.min(PRED),np.max(PRED)))
    print('PRED labels: {}'.format(np.unique(PRED)))

    # remove offset    
    dim_offset = np.asarray(dim_pred - dim_data, dtype=int)
    print('Remove offset: {} to {} => {} ...'.format(dim_pred,dim_data,dim_offset))
    dim_end = np.asarray(dim_pred - dim_offset, dtype=int)
    PRED  =  PRED[0:dim_end[0],0:dim_end[1],0:dim_end[2]]
    TILES = TILES[0:dim_end[0],0:dim_end[1],0:dim_end[2]]
    PROBS = PROBS[::,0:dim_end[0],0:dim_end[1],0:dim_end[2]]
    # ignore voxels with zero mask
    PRED[MASKin==0.0] = 0
    # probs
    for p in range(0,np.shape(PROBS)[0]):
        probs = np.squeeze(PROBS[p,:,:,:])
        probs[MASKin==0.0] = 0
        PROBS[p,:,:,:] = probs
    print('\tsize data {},\n\t size prediction {},\n\t size tiles {},\n\t size probs {}.'.format(dim_data,np.shape(PRED),np.shape(TILES),np.shape(PROBS)))
    print('save ({}) with outprefix at {}'.format(np.shape(PRED),outprefix))
    out_prediction_file = outprefix+'--PRED.nii.gz'
    nib.save( nib.Nifti1Image(np.transpose(PRED,(2,1,0)),np.eye(4)) , out_prediction_file)    
    nib.save( nib.Nifti1Image(np.transpose(TILES,(2,1,0)),np.eye(4)) , outprefix+'--TILES.nii.gz')    
    nib.save( nib.Nifti1Image(np.transpose(PROBS,(3,2,1,0)),np.eye(4)), outprefix+'--PROBS.nii.gz')  
    
    if EXTRACT_FEATURES:
        FEATS = FEATS[::,0:dim_end[0],0:dim_end[1],0:dim_end[2]]
        for f in range(0,np.shape(FEATS)[0]):
            feats = np.squeeze(FEATS[f,:,:,:])
            feats[MASKin==0.0] = 0
            FEATS[p,:,:,:] = feats
        nib.save( nib.Nifti1Image(np.transpose(FEATS,(3,2,1,0)),np.eye(4)), outprefix+'--FEATS.nii.gz')      
        
    return out_prediction_file, np.transpose(PRED,(2,1,0))

def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

#Visualization
def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    
    plt.figure()
    for i in range(0, len(scale_lst)):
        s=plt.subplot(1,5,i+1)
        plt.imshow(1-scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()
    
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
        
def predict_file(proto_text,trained_model,infile,output_root,device=0): 
    print(100*'=')
    print(100*'=')
    print('DEPLOY CNN...')
    base_name = os.path.splitext(os.path.basename(infile))[0]
    outprefix = output_root+'/'+base_name+'/'+base_name
    image = infile.replace('.h5','--data.nii.gz')        
    mask = infile.replace('.h5','--mask.nii.gz')        
    print(' with image \t{}\n and mask \t{}'.format(image,mask))    
    print('  save to {}*'.format(outprefix))        
    out_prediction_file, PRED = deploy(proto_text,trained_model,image,mask,outprefix,device)
    return out_prediction_file, PRED

def resample_and_save_raw(PRED,orig_size,orig_spacing,raw_name,method="nearest"):
    pred_size = np.shape(PRED)
    xi = np.linspace(0,pred_size[0]-1,orig_size[0])
    yi = np.linspace(0,pred_size[1]-1,orig_size[1])
    zi = np.linspace(0,pred_size[2]-1,orig_size[2])
    XI, YI, ZI = np.meshgrid(xi, yi, zi)            
    print('Interp3 of PRED shape {} to Image shape {} and save to {}'.format(pred_size,orig_size,raw_name))
    PRED = interp3([0,pred_size[0]],[0,pred_size[1]],[0,pred_size[2]],PRED,\
                   XI, YI, ZI,\
                   method=method)     
    mori.write_mori(np.asarray(np.transpose(PRED,(1,0,2)),dtype=np.uint8),orig_spacing,raw_name,use_gzip=True)    
        
def deploy_cascade(img_file,label_file=None,output_root='/tmp',device=0):
    basename = os.path.splitext(os.path.basename(img_file))[0]
    output_root = os.path.join(output_root,basename)
    output_root1 = os.path.join(output_root,basename+'_stage1')   
    h5file1 = os.path.join(output_root1,basename+'_data1.h5')   
    output_root2 = os.path.join(output_root,basename+'_stage2')
    h5file2 = os.path.join(output_root2,basename+'_data2.h5')   
    
    if not os.path.isdir(output_root1):
        os.makedirs(output_root1)
    if not os.path.isdir(output_root2):    
        os.makedirs(output_root2)
        
    orig_size, orig_spacing = read_image_info(img_file)
        
    # Stage 1
    start = time.time()        
    convert_image_and_label_to_h5(img_file,label_file,h5file1,None,0,win_min1,win_max1,ZERO_MEAN1,NORM1)    
    prediction_file1, PRED = predict_file(proto_text1,trained_model1,h5file1,output_root1,device)
    end = time.time()
    time1 = end - start
    # resample to original size and save in mori formal
    raw_prediction_file1 = prediction_file1.replace('.nii.gz','.header')
    resample_and_save_raw(PRED,orig_size,orig_spacing,raw_prediction_file1,method="nearest")
    
    # Stage 2
    start = time.time()        
    convert_image_and_label_to_h5(img_file,label_file,h5file2,prediction_file1,DILATE_MASK_TO_INCLUDE,win_min2,win_max2,ZERO_MEAN2,NORM2)
    prediction_file2, PRED = predict_file(proto_text2,trained_model2,h5file2,output_root2,device)
    end = time.time()
    time2 = end - start
    # resample to original size and save in mori formal
    raw_prediction_file2 = prediction_file2.replace('.nii.gz','.header')
    resample_and_save_raw(PRED,orig_size,orig_spacing,raw_prediction_file2,method="nearest")
    
    print('Final result saved at (nifti)         : {}'.format(prediction_file2))
    print('Final result saved at (orig. size raw): {}'.format(raw_prediction_file2))
    end = time.time()
    print('Stage1 time: {:.1f} seconds (= {:.1f} minutes).'.format(time1,time1/60))
    print('Stage2 time: {:.1f} seconds (= {:.1f} minutes).'.format(time2,time2/60))
    print('Total time: {:.1f} seconds (= {:.1f} minutes).'.format(time1+time2,(time1+time2)/60))
    return raw_prediction_file2

def get_visceral_radlexIDs():
    radlexIDs = {}
    radlexIDs['29663'] = ['Left Kidney',                           1]
    radlexIDs['29662'] = ['Right Kidney',                          2]
    radlexIDs['86']    = ['Spleen',                                3]
    radlexIDs['58']    = ['Liver',                                 4]
    radlexIDs['1326']  = ['Left Lung',                             5]
    radlexIDs['1302']  = ['Right Lung',                            6]
    radlexIDs['237']   = ['Urinary bladder',                       7]
    radlexIDs['40358'] = ['Muscle body of left rectus abdominis',  8]
    radlexIDs['40357'] = ['Muscle body of right rectus abdominis', 9]
    radlexIDs['29193'] = ['Lumbar Vertebra 1',                    10]
    radlexIDs['7578']  = ['Thyroid',                              11]
    radlexIDs['170']   = ['Pancreas',                             12]
    radlexIDs['32248'] = ['Left psoas major muscle',              13]
    radlexIDs['32249'] = ['Right psoas major muscle',             14]
    radlexIDs['187']   = ['Gallbladder (without ductus',          15]
    radlexIDs['2473']  = ['Sternum',                              16]
    radlexIDs['480']   = ['Aorta',                                17]
    radlexIDs['1247']  = ['Trachea',                              18]
    radlexIDs['30325'] = ['Left Adrenal Gland',                   19]
    radlexIDs['30324'] = ['Right Adrenal Gland',                  20]    
    return radlexIDs

def get_visceral_class_label(radlexID):
    radlexIDs = get_visceral_radlexIDs()
    return radlexIDs.get(radlexID)
    
from __future__ import absolute_import
import sys
import os
#import glob
from PIL import Image
import numpy as np
from six.moves import range
import random
import cPickle as pickle      
#from nifti import *    
#import nrrd 
import hickle               

def get_colors(N):
    np.random.seed(0)
    return np.random.rand(N,3)    
    
def get_markers(N)    :
    # matplotlib markers
    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
    markers = []
    i = 0
    while len(markers) < N:
        if i<len(filled_markers):
            markers.append(filled_markers[i])
            i += 1
        else:
            i = 0
    return markers

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
               
def get_label_names_abdomen():
    label_names = [('label255.', 'neg',255),
                   ('label1.','spleen',1),
                   ('label2.','right kidney',2),
                   ('label3.','left kidney',3),
                   ('label4.','gallbladder',4),
                   ('label5.','esophagus',5),
                   ('label6.','liver',6),
                   ('label7.','stomach',7),
                   ('label8.','aorta',8),
                   ('label9.','IVC',9),
                   ('label10.','portal and splenic vein',10),
                   ('label11.','pancreas',11),
                   ('label12.','right adrenal',12),
                   ('label13.','left adrenal',13)]
    return label_names
               
def get_organs_abdomen():       
    labels = get_label_names_abdomen()        
    organs = []
    for label in labels:
        organs.append(label[1])
    return organs
               
def recursive_glob(searchroot='.', searchstr=''):
    if not os.path.isdir(searchroot):
        raise ValueError('No such directory: {}'.format(searchroot))
    print "search for {0} in {1}".format(searchstr,searchroot)
    f = [os.path.join(rootdir, filename)
        for rootdir, dirnames, filenames in os.walk(searchroot)
        for filename in filenames if searchstr in filename]
    f.sort()
    return f        
    
def recursive_glob2(searchroot='.', searchstr1='', searchstr2=''):
    if not os.path.isdir(searchroot):
        raise ValueError('No such directory: {}'.format(searchroot))
    print "search for {} and {} in {}".format(searchstr1,searchstr2,searchroot)
    f = [os.path.join(rootdir, filename)
        for rootdir, dirnames, filenames in os.walk(searchroot)
        for filename in filenames if (searchstr1 in filename and searchstr2 in filename)]
    f.sort()
    return f            

def find_files(rootdirs=['.'], searchstr=''):
    if isinstance(rootdirs,str):
        return recursive_glob(rootdirs, searchstr)
    elif isinstance(rootdirs,list):
        fs = []
        for rootdir in rootdirs:
            fs.extend( recursive_glob(rootdir, searchstr) )
        return fs
    else:
        print("find_files expects string or list of strings as input!")
        raise TypeError

def load_images(fs,label_names,N, Nchannels, ImSize):
    label_counts = [0] * label_names.__len__()        
    
    if N<0:
        N = fs.__len__()
    
    X = np.ndarray(shape=(N, Nchannels, ImSize, ImSize), dtype = np.uint8)
    Y = np.ndarray(shape=(N, 1), dtype = np.uint8)
    for i in range(0,N):
        # get label
        curr_path,curr_filename=os.path.split(fs[i])
        label_found = False
        for label_idx, label_name in enumerate(label_names):
            if label_name[0] in curr_filename:
                label = label_idx
                label_found = True
                label_counts[label_idx] = label_counts[label_idx] + 1

        if not label_found:
            print "Error: could not find label in {} -> skip".format(curr_filename)
            sys.exit( 1 )  
        
        im = Image.open(fs[i], 'r')
        X[i,:,:,:] = np.swapaxes(im,0,2)
                    
        #print extension
        Y[i] = label    
        
        # progress
        if i%10000 is 0:
            print "{0} of {1}: {2}".format(i, N, fs[i])
        
    total = 0
    for label_idx, label_name in enumerate(label_names):
        print " {}: {}".format(label_name,label_counts[label_idx])        
        total += label_counts[label_idx]
        
    print " Total: {0}".format( total )    
        
    return (X, Y, fs[0:N])

def make_data(train_path,test_path,Nchannels, ImSize, Ntrain=-1,Ntest=-1):   
    
    label_names = get_label_names_abdomen()    
    
    print "There are {} label names:".format(label_names.__len__())
    for label_name in label_names:
        print "    {}".format(label_name)
    
    # Training data
    print "Training data"
    # Find files
    fs = find_files(train_path,'.png')
    print " Randomly shuffle names..."
    random.shuffle(fs)    
    (X_train, Y_train, Files_train) = load_images(fs,label_names,Ntrain, Nchannels, ImSize)

    # Testing data
    print "Test data"
    # Find files
    fs = find_files(test_path,'.png')
    print " Randomly shuffle names..."
    random.shuffle(fs)
    (X_test, Y_test, Files_test) = load_images(fs,label_names,Ntest, Nchannels, ImSize)
        
    print "success"
    
    return (X_train, Y_train), (X_test, Y_test), (Files_train, Files_test) 
 
def load_volumes(fs,label_names, Nchannels, ImSize, FOR_3DCONV):
    label_counts = [0] * label_names.__len__()        
    
    N = len(fs)

    if N>0:
        nim = NiftiImage(fs[0]) # find data type
        data_type = nim.data.dtype
    else:
        data_type = 'uint8'
        
    print(' Load volumes with data type: {}'.format(data_type))
    X = np.ndarray(shape=(N, Nchannels, ImSize, ImSize, 1), dtype = data_type)
    Y = np.ndarray(shape=(N, 1), dtype = np.uint8)            
    for i in range(0,N):  
        # get label
        curr_path,curr_filename=os.path.split(fs[i])
        label_found = False
        for label_idx, label_name in enumerate(label_names):
            if label_name[0] in curr_filename:
                label = label_idx
                label_found = True
                label_counts[label_idx] = label_counts[label_idx] + 1

        if not label_found:
            print "Error: could not find label in {} -> skip".format(curr_filename)
            sys.exit( 1 )  
        
        nim = NiftiImage(fs[i])
        X[i,:,:,:,0] = nim.data
                    
        #print extension
        Y[i] = label    
        
        # progress
        if i%10000 is 0:
            print "{0} of {1}: {2}".format(i, N, fs[i])
        
    total = 0
    for label_idx, label_name in enumerate(label_names):
        print " {}: {}".format(label_name,label_counts[label_idx])        
        total += label_counts[label_idx]
    print " Total: {0}".format( total )    
    
	 # Need to order the dataset as per the convolutional layer used.
    # As conv2d3d.conv3d doesnot support operation on a cpu, we are using nnet.conv3d for training on cpu.
    # The order of inputs are diferent for both implementation. Hence, the follwing channel shuffling. 

    #(http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d)
    #0 ; batch_size
    #1 ; stack size, number of channels(z) in 3D data
    #2 ; image row size
    #3 ; image column size
    #4 ; 4th dimension, set to  1 for one channel in 3D data
    if FOR_3DCONV:
        if FOR_GPU:
            X = X.transpose(0, 1, 4, 2, 3) # as required by conv2d3d.Conv3d
        else: #'cpu'
            X = X.transpose(0, 2, 3, 1, 4) # as required by conv3D
    else:
        X = X.squeeze()
                
    return (X, Y, fs[0:N]) 
 
# FOR_3DCONV if True, data is prepared for keras convolution3D(http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d)
def make_data_3D(out_hickle_file,train_path,test_path,Nchannels, ImSize, Ntrain=-1,Ntest=-1,Nbatches=1,FOR_3DCONV=False):   
    if not out_hickle_file.endswith('.hkl'): # check extension
        raise ValueError('out_hickle_file must have extension .hkl !')
        
    if not os.path.isdir(os.path.dirname(out_hickle_file)):
        os.makedirs(os.path.dirname(out_hickle_file))
        
    label_names = get_label_names_abdomen()    
    
    print "There are {} label names:".format(label_names.__len__())
    for label_name in label_names:
        print "    {}".format(label_name)
    
    # Find files
    fs_train = find_files(train_path,'.nii.gz')
    print " Randomly shuffle training names..."
    random.shuffle(fs_train)    
    fs_test = find_files(test_path,'.nii.gz')
    
    if Ntrain<0:
        Ntrain = len(fs_train)    
    if Ntest<0:
        Ntest = len(fs_test)    
        
    train_step = Ntrain/Nbatches
    if train_step<1 and train_step is not 0:
        raise ValueError('Training batch step cannot be smaller 1!')
    test_step = Ntest/Nbatches
    if test_step<1 and test_step is not 0:
        raise ValueError('Testing batch step cannot be smaller 1!')    
        
    out_hickle_batches = Nbatches * [None]    
    for b in range(0,Nbatches):
        # Training data
        print(40*'-')
        print "Training data"
        print(40*'-')        
        i0 = b*train_step
        i1 = b*train_step+train_step-1
        print(' Batches {} of {} in range [{}, {}]...'.format(b+1,Nbatches,i0,i1))
        (X_train, Y_train, Files_train) = load_volumes(fs_train[i0:i1],label_names, Nchannels, ImSize, FOR_3DCONV)
    
        # Testing data
        print(40*'-')
        print "Test data"
        print(40*'-')    
        i0 = b*test_step
        i1 = b*test_step+test_step
        print(' Batches {} of {} in range [{}, {}]...'.format(b+1,Nbatches,i0,i1))        
        (X_test, Y_test, Files_test) = load_volumes(fs_test[i0:i1],label_names, Nchannels, ImSize, FOR_3DCONV)
    
        print(40*'-')        
        print "success"
        print(40*'-')
    
        print("Save hickle batch {} of {}...".format(b+1,Nbatches))
        out_hickle_batches[b] = out_hickle_file.replace('.hkl','_'+str(b)+'.hkl')
        hickle.dump( (X_train, Y_train, X_test, Y_test, Files_train, Files_test), out_hickle_batches[b], mode='w')       
        print "Saved hickle at: {}".format(out_hickle_batches[b])    
        
    return out_hickle_batches
   
def load_data(path):
     print('Load {}'.format(path))
     filename, file_extension = os.path.splitext(path)
     if '.pkl' in file_extension:
         return load_data_pkl(path)
     elif '.hkl' in file_extension:
         return load_data_hkl(path)         
     elif '.nrrd' in file_extension:
         return load_data_nrrd(path)                  
     else:  
         print("load_data: Extension not supported " + path + "-> ext: " + file_extension)
         raise ValueError
    
def load_data_pkl(path):
    D = pickle.load( open( path, "rb" ) )
    X_train = D[0][0] 
    Y_train = D[0][1]
    X_test = D[1][0] 
    Y_test = D[1][1]    
    
    if D.__len__()>2:
        Files_train = D[2][0]
        Files_test = D[2][1]
    else:
        Files_train = []
        Files_test = []

    return (X_train, Y_train), (X_test, Y_test), (Files_train, Files_test)     
    
   
def load_data_hkl(path):
    D = hickle.load( path )
    X_train = D[0] 
    Y_train = D[1]
    X_test = D[2] 
    Y_test = D[3]    
    
    if D.__len__()>4:
        Files_train = D[4]
        Files_test = D[5]
    else:
        Files_train = []
        Files_test = []

    return (X_train, Y_train), (X_test, Y_test), (Files_train, Files_test) 

def load_data_nrrd(path):
    # only works for testing data
    X_train = []
    Y_train = []
    Files_train = []
    
    X_test = nrrd.read(path)[0] 
    Meta = nrrd.read(path.replace('_data','_meta'))[0]
    Files_test = Meta # same information as used to be in filenames
    # convert label values to label index
    labels = get_label_names_abdomen()
    Y_test = Meta[:,0]
    for i in range(0,Meta.shape[0]):
        for l in range(0,len(labels)):
            if Meta[i,0] == labels[l][2]:
                Y_test[i] = l        

    return (X_train, Y_train), (X_test, Y_test), (Files_train, Files_test) 
        
def load_activation_data(path):

    try: 
        a = hickle.load( path )
    
        Files    = a["Files"]
        Y        = a["Y"]
        Y_catmat = a["Y_catmat"]
        activations = a["activations"]    
    except ValueError:
        print "Error reading pickle file (corrupted?): {0}".format(path)       
        Files    = []
        Y        = []
        Y_catmat = []
        activations = []            
        
    return Files, activations, Y, Y_catmat
    
def getLocations(Files, offset=[0.0, 0.0, 0.0], norm_fac=[1.0, 1.0, 1.0]):
    N = len(Files)
    print "get locations from {0} files".format(N)
    loc = np.zeros((N,3))
    for i in range(0,N):
        f = Files[i]
        path, name = os.path.split(f)
        #print(name)
        sub = name[name.find('_x')+2:]
        x = float(sub[:sub.find('_')])
        sub = name[name.find('_y')+2:]
        y = float(sub[:sub.find('_')])
        sub = name[name.find('_z')+2:]
        z = float(sub[:sub.find('_')])        
        #print([x, y, z])
        loc[i][:] = [norm_fac[0]*(x+offset[0]), norm_fac[1]*(y+offset[1]), norm_fac[2]*(z+offset[2])]
        
        # progress
        if i%10000 is 0:
            print "{0} of {1}: {2} -> {3}".format(i, N, name, loc[i][:])        
    return loc
        
def windowing(x,lower,upper,desiredMin,desiredMax):

    m = (desiredMax-desiredMin)/(upper-lower);
    t = desiredMax - m*upper;

    y = m*x + t;
    
    y[y<desiredMin] = desiredMin;
    y[y>desiredMax] = desiredMax;
    
    #print (" windowing between [{}, {}] to [{}, {}] using y = {}*x + {}".format(lower,upper,desiredMin,desiredMax,m,t))
    
    return y
        

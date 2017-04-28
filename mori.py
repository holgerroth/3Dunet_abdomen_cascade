# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:26:00 2017

@author: rothhr

###################################
Usage examples:

# read raw header
hdr = mori.read_mori_header(filename)

# read raw image
I, hdr = mori.read_mori(filename,dtype)

# write raw image
mori.write_mori(I,spacing,filname,use_gzip=True)    
"""



import os
import numpy as np
import gzip

def read_mori_header(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    hdr = {}    
    for line in lines:
        if 'OrgFile' in line:
            hdr['OrgFile'] = line[line.find(':')+1::].replace(' ','')
        if 'SizeX' in line:
            hdr['SizeX'] = int(line[line.find(':')+1::])
        if 'SizeY' in line:
            hdr['SizeY'] = int(line[line.find(':')+1::])
        if 'SizeZ' in line:
            hdr['SizeZ'] = int(line[line.find(':')+1::])            
        if 'PitchX' in line:
            hdr['PitchX'] = float(line[line.find(':')+1::])            
        if 'PitchY' in line:
            hdr['PitchY'] = float(line[line.find(':')+1::])            
        if 'PitchZ' in line:
            hdr['PitchZ'] = float(line[line.find(':')+1::])       
    hdr['size'] = np.asarray([hdr['SizeX'],hdr['SizeY'],hdr['SizeZ']])            
    hdr['spacing'] = np.asarray([hdr['PitchX'],hdr['PitchY'],hdr['PitchZ']])
    return hdr        

def read_mori_raw(filename,dtype):
    basename = os.path.basename(filename)
    if '.gz' in basename:
        with gzip.open(filename, 'rb') as f:
            file_content = f.read()
        I = np.frombuffer(file_content,dtype)
    else:
        I = np.fromfile(filename,dtype)
    I.flags['WRITEABLE'] = True
    NZ = np.size(I)/(512*512) # assuem Mori raw with 512x512xNZ shape or mhd
    I = np.reshape(I,(NZ,512,512))
    I = np.transpose(I,(2,1,0))
    I = I[::-1,::-1,::-1] # reverse
    if 'u2' in dtype or dtype==np.uint16:
        print('[WARNING] trying to set non-reconstructed scanner values to zero!!!!!!!!!!!!!')
        I[I>=np.iinfo(dtype).max*0.9] = 0 # set non-reconstructed values to zero (useful for some scanners)    
    return I
        
def read_mori(filename,dtype):
    if dtype is None:
        raise ValueError('You need to specifiy the data type for mori raw format, \ e.g. ">u1" (uint8), ">u2" (uint16),...')
    if '.header' in filename:          
        hdr = read_mori_header(filename)
        rawfile = os.path.join(os.path.split(filename)[0],hdr['OrgFile'])
        I = read_mori_raw(rawfile,dtype)
    else:
        hdr = None
        I = read_mori_raw(filename,dtype)
    return I, hdr
        
def write_mori(I,spacing,filename,use_gzip=True):           
    I = I[::-1,::-1,::-1] # reverse
    
    if I.dtype == np.uint8:
        raw_ext = '.uc_raw'
    else:
        raw_ext = '.raw'
        
    filename = os.path.splitext(filename)[0] # remove extensions given
    # write raw    
    if use_gzip:    
        out_rawfilename = filename+raw_ext+'.gz'
        with gzip.open(out_rawfilename, 'wb') as f:
            f.write(I.tobytes('F')) # Fortran order seems to work
    else:
        out_rawfilename = filename+raw_ext
        with open(out_rawfilename, 'wb') as f:
            f.write(I.tobytes('F'))
        
    # write header
    out_hdrfilename = filename+'.header'
    with open(out_hdrfilename,'w') as f:
        hdr_fields = get_mori_header_fields()
        for field in hdr_fields: # \r\n
            if 'OrgFile' in field:
                f.write('{} {}\r\n'.format(field,os.path.split(out_rawfilename)[1]))                
            elif 'SizeX' in field:
                f.write('{} {}\r\n'.format(field,np.shape(I)[0]))
            elif 'SizeY' in field:
                f.write('{} {}\r\n'.format(field,np.shape(I)[1]))
            elif 'SizeZ' in field:
                f.write('{} {}\r\n'.format(field,np.shape(I)[2]))
            elif 'PitchX' in field:
                f.write('{} {:.6f}\r\n'.format(field,spacing[0]))
            elif 'PitchY' in field:
                f.write('{} {:.6f}\r\n'.format(field,spacing[1]))
            elif 'PitchZ' in field:
                f.write('{} {:.6f}\r\n'.format(field,spacing[2]))
            else:
                f.write('{} \r\n'.format(field))
    return out_hdrfilename                
                
def get_mori_header_fields():
    hdr_fields = [];
    hdr_fields.append('OrgFile       :')
    hdr_fields.append('MarkFile1     :') 
    hdr_fields.append('SizeX         :')
    hdr_fields.append('SizeY         :')
    hdr_fields.append('SizeZ         :')
    hdr_fields.append('PitchX        :')
    hdr_fields.append('PitchY        :')
    hdr_fields.append('PitchZ        :')
    hdr_fields.append('Thickness     : 1.000000')
    hdr_fields.append('ImagePositionBegin   :')
    hdr_fields.append('ImagePositionEnd   :')
    hdr_fields.append('Orientation   : LPF')
    hdr_fields.append('PatientID     :')
    hdr_fields.append('Hospital      :')
    hdr_fields.append('Exp_Date_Year : 0')
    hdr_fields.append('Exp_Date_Month: 0')
    hdr_fields.append('Exp_Date_Day  : 0')
    hdr_fields.append('Trs_Date_Year : 0')
    hdr_fields.append('Trs_Date_Month: 0')
    hdr_fields.append('Trs_Date_Day  : 0')
    hdr_fields.append('KVP           :')
    hdr_fields.append('AMP           :')
    hdr_fields.append('KernelFunction :')
    hdr_fields.append('ModelName     :')
    hdr_fields.append('PatientPosition    : HFS')
    hdr_fields.append('PatientOrientation : L\\P')
    hdr_fields.append('ImageOrientation  : 1.000000\\0.000000\\0.000000\\0.000000\\1.000000\\0.000000')
    hdr_fields.append('ImagePosition  : None')
    hdr_fields.append('StudyDate     :')
    hdr_fields.append('SeriesData    :') 
    hdr_fields.append('AcquisitionDate :')
    hdr_fields.append('Comment1 :')
    hdr_fields.append('Comment2 :')
    hdr_fields.append('Comment3 :')
    hdr_fields.append('Comment4 :')
    hdr_fields.append('Comment5 :')
    hdr_fields.append('Comment6 :')
    hdr_fields.append('Comment7 :')
    hdr_fields.append('Comment9 :')
    hdr_fields.append('Comment9 :')
    hdr_fields.append('Comment10 :')
    hdr_fields.append('Comment11 :')
    hdr_fields.append('Comment12 :')
    hdr_fields.append('Comment13 :')
    hdr_fields.append('Comment14 :')
    hdr_fields.append('Comment15 :')
    hdr_fields.append('Comment16 :')
    hdr_fields.append('Comment17 :')
    hdr_fields.append('Comment18 :')
    hdr_fields.append('Comment19 :')
    hdr_fields.append('Comment20 :')
    return hdr_fields
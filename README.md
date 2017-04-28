# 3Dunet_abdomen_cascade

This repository provides the code and models files for multi-organ segmentation in abdominal CT using cascaded 3D U-Net models. The models are described in:

"Hierarchical 3D fully convolutional networks for multi-organ segmentation"
Holger R. Roth, Hirohisa Oda, Yuichiro Hayashi, Masahiro Oda, Natsuki Shimizu, Michitaka Fujiwara, Kazunari Misawa, Kensaku Mori
https://arxiv.org/abs/1704.06382

This work is based on the open-source implementation of 3D U-Net: https://lmb.informatik.uni-freiburg.de/resources/opensource/unet.en.html
We thank the authors for providing their implementation.

Olaf Ronneberger, Philipp Fischer & Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351, 234--241, 2015	DOI  Code
and
Özgün Çiçek, Ahmed Abdulkadir, S. Lienkamp, Thomas Brox & Olaf Ronneberger. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9901, 424--432, Oct 2016

3D U-Net is based on Caffe. To compile, follow the Caffe instructions:
http://caffe.berkeleyvision.org/installation.html#prequequisites

To run the segmentation algorithm on a new case use:
python run_full_cascade_deploy.py
Note, please update the paths in run_full_cascade_deploy.py

You might have to add a -2000 offset to
win_min/max1/2 in deploy_cascade.py if your images are in Hounsfield units.

For training, please follow the 3D U-Net instruction.
prepare_data.py can be useful for converting nifti images and label images to h5 containers which can be read by caffe.

Please contact Holger Roth (rothhr@mori.m.is.nagoya-u.ac.jp) for any questions.

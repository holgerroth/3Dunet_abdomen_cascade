#The names should be adapted accordingly.
CAFFE_BIN=../caffe_unet_3D_v1.0_patch/caffe/build/tools/caffe.bin

HDF5_DISABLE_VERSION_CHECK=1 time ${CAFFE_BIN} train --solver=solver.prototxt 2>&1| tee log/3dUnet_Visceral_with_BN-`date +%F_%R`.log

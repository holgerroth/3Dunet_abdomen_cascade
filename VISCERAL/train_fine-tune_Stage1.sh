#The names should be adapted accordingly.
CAFFE_BIN=../caffe_unet_3D_v1.0_patch/caffe/build/tools/caffe.bin

WEIGHTS=../snapshot-Stage2/3dUnet_Abdomen_with_BN_normed-Stage2_iter_115000.caffemodel # fine-tune from abdomen

HDF5_DISABLE_VERSION_CHECK=1 time ${CAFFE_BIN} train --solver=solver_Stage1.prototxt -weights ${WEIGHTS} -sighup_effect snapshot -sigint_effect snapshot 2>&1| tee log/3dUnet_Visceral_with_BN_abd-finetune-Stage1`date +%F_%R`.log

#The names should be adapted accordingly.
CAFFE_BIN=../caffe_unet_3D_v1.0_patch/caffe/build/tools/caffe.bin

SOLVERSTATE=snapshots-Stage2/3dUnet_Visceral_with_BN_abd-finetune-Stage2_iter_40806.solverstate

HDF5_DISABLE_VERSION_CHECK=1 time ${CAFFE_BIN} train --solver=solver_Stage2.prototxt -snapshot ${SOLVERSTATE} -sighup_effect snapshot -sigint_effect snapshot 2>&1| tee log/3dUnet_Visceral_with_BN_abd-finetune-Stage1`date +%F_%R`.log

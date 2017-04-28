#include <vector>

#include "caffe/layers/apply_deformation_layer.hpp"
#include "caffe/util/vector_helper.hpp"

namespace caffe {

template <typename Dtype>
void ApplyDeformationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(1, top.size());
  Reshape(bottom, top);
}

template <typename Dtype>
void ApplyDeformationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom.size(), 2);
  CHECK_GE(bottom[1]->num_axes(), 4);
  CHECK_LE(bottom[1]->num_axes(), 5);
  CHECK_GE(bottom[1]->shape(-1), 2);
  CHECK_LE(bottom[1]->shape(-1), 3);

  const ApplyDeformationParameter& param =
      this->layer_param_.apply_deformation_param();

  std::string shapeBlobName = param.output_shape_from();

  if(shapeBlobName == "") {
    // get output shape from deformation field
    std::vector<int> outShape = bottom[0]->shape();
    outShape[2] = bottom[1]->shape(1);
    outShape[3] = bottom[1]->shape(2);
    if( bottom[0]->num_axes() == 5 && bottom[1]->num_axes() == 5) {
      outShape[4] = bottom[1]->shape(3);
    }
    top[0]->Reshape( outShape);
  } else {
    // search for blob in list all blob names
    // (the blob_by_name map is not initialized yet)
    //
    //    std::cout << "searching for " << shapeBlobName << std::endl;
    int blobIdx = -1;
    for( int i = 0; i < this->parent_net()->blob_names().size(); ++i) {
      // std::cout << "blob " << i << " with name " << this->parent_net()->blob_names()[i] << std::endl;
      if(  this->parent_net()->blob_names()[i] == shapeBlobName) {
        blobIdx = i;
        break;
      }
    }
    if( blobIdx == -1) {
      LOG(WARNING) << "output_shape_from: Unknown blob name " << shapeBlobName;
      CHECK( false);
    }
    // get output shape from other blob
    // find blob by name
    const shared_ptr<Blob<Dtype> > shapeBlob =
        this->parent_net()->blobs()[blobIdx];
    std::vector<int> outShape = shapeBlob->shape();
    outShape[0] = bottom[0]->shape(0);
    outShape[1] = bottom[0]->shape(1);
    top[0]->Reshape( outShape);
  }
  if (bottom[1]->num_axes() == 4) {
    n_spatial_axes_ = 2;
    n_deform_comps_ = 2;
  } else {
    n_spatial_axes_ = 3;
    n_deform_comps_ = bottom[1]->shape(4);
  }
}

template <typename Dtype>
void ApplyDeformationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const ApplyDeformationParameter& param =
      this->layer_param_.apply_deformation_param();

  const Dtype* in = bottom[0]->cpu_data();
  int inNb  = bottom[0]->shape(0);
  const int inNc  = bottom[0]->shape(1);
  const int inNz  = (n_spatial_axes_ == 2)? 1 : bottom[0]->shape(-3);
  const int inNy  = bottom[0]->shape(-2);
  const int inNx  = bottom[0]->shape(-1);

  const Dtype* def = bottom[1]->cpu_data();
  const int defNb  = bottom[1]->shape(0);
  const int defNz  = (n_spatial_axes_ == 2)? 1 : bottom[1]->shape(-4);
  const int defNy  = bottom[1]->shape(-3);
  const int defNx  = bottom[1]->shape(-2);

  Dtype* out      = top[0]->mutable_cpu_data();
  const int outNb = top[0]->shape(0);
  const int outNc = top[0]->shape(1);
  const int outNz = (n_spatial_axes_ == 2)? 1 : top[0]->shape(-3);
  const int outNy = top[0]->shape(-2);
  const int outNx = top[0]->shape(-1);

  if (n_deform_comps_ == 2
      && param.interpolation() == "linear"
      && param.extrapolation() == "mirror") {
    transform2D(
        in,   inNb,  inNc,  inNz,  inNy,  inNx,
        def, defNb,        defNz, defNy, defNx,
        out, outNb, outNc, outNz, outNy, outNx, linear2D_mirror<Dtype>);
  } else if (n_deform_comps_ == 2
             && param.interpolation() == "linear"
             && param.extrapolation() == "zero") {
    transform2D(
        in,   inNb,  inNc,  inNz,  inNy,  inNx,
        def, defNb,        defNz, defNy, defNx,
        out, outNb, outNc, outNz, outNy, outNx, linear2D_zeropad<Dtype>);
  } else if (n_deform_comps_ == 2
             && param.interpolation() == "nearest"
             && param.extrapolation() == "mirror") {
    transform2D(
        in,   inNb,  inNc,  inNz,  inNy,  inNx,
        def, defNb,        defNz, defNy, defNx,
        out, outNb, outNc, outNz, outNy, outNx, nearest2D_mirror<Dtype>);
  } else if (n_deform_comps_ == 2
             && param.interpolation() == "nearest"
             && param.extrapolation() == "zero") {
    transform2D(
        in,   inNb,  inNc,  inNz,  inNy,  inNx,
        def, defNb,        defNz, defNy, defNx,
        out, outNb, outNc, outNz, outNy, outNx, nearest2D_zeropad<Dtype>);
  } else if (n_deform_comps_ == 3
      && param.interpolation() == "linear"
      && param.extrapolation() == "mirror") {
    transform3D(
        in,   inNb,  inNc,  inNz,  inNy,  inNx,
        def, defNb,        defNz, defNy, defNx,
        out, outNb, outNc, outNz, outNy, outNx, linear3D_mirror<Dtype>);
  } else if (n_deform_comps_ == 3
             && param.interpolation() == "linear"
             && param.extrapolation() == "zero") {
    transform3D(
        in,   inNb,  inNc,  inNz,  inNy,  inNx,
        def, defNb,        defNz, defNy, defNx,
        out, outNb, outNc, outNz, outNy, outNx, linear3D_zeropad<Dtype>);
  } else if (n_deform_comps_ == 3
             && param.interpolation() == "nearest"
             && param.extrapolation() == "mirror") {
    transform3D(
        in,   inNb,  inNc,  inNz,  inNy,  inNx,
        def, defNb,        defNz, defNy, defNx,
        out, outNb, outNc, outNz, outNy, outNx, nearest3D_mirror<Dtype>);
  } else if (n_deform_comps_ == 3
             && param.interpolation() == "nearest"
             && param.extrapolation() == "zero") {
    transform3D(
        in,   inNb,  inNc,  inNz,  inNy,  inNx,
        def, defNb,        defNz, defNy, defNx,
        out, outNb, outNc, outNz, outNy, outNx, nearest3D_zeropad<Dtype>);
  } else {
    CHECK(0) << "unsuppported combination: n_deform_comps_=" << n_deform_comps_
                 << ", interpolation=" << param.interpolation()
                 << ", exrapolation=" << param.extrapolation();
  }

}

template <typename Dtype>
void ApplyDeformationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
 }

INSTANTIATE_CLASS(ApplyDeformationLayer);

REGISTER_LAYER_CLASS(ApplyDeformation);

}  // namespace caffe

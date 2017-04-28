#ifndef CREATE_DEFORMATION_LAYER_HPP_
#define CREATE_DEFORMATION_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

/**
 * @brief Provides random deformation fields to the Net. Use this as
 * input with ApplyDeformationLayer for data augmentation
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class CreateDeformationLayer : public Layer<Dtype> {
 public:
  explicit CreateDeformationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CreateDeformation"; }
  virtual inline int MinBottomBlobs() const { return 0; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  Dtype* create_bspline_kernels(int nb);

  void cubic_bspline_interpolation(
    const Dtype* in, int in_n_lines, int in_n_elem,
    int in_stride_lines, int in_stride_elem,
    Dtype* out, int out_n_elem, int out_stride_lines, int out_stride_elem,
    const Dtype* b0123, int nb) ;

  int batch_size_;
  int n_spatial_axes_;
  int n_deform_comps_;

  bool do_elastic_trafo_;
  int grid_spacing_x_;
  int grid_spacing_y_;
  int grid_spacing_z_;
  Dtype* bkernel_x_;
  Dtype* bkernel_y_;
  Dtype* bkernel_z_;
  Dtype* rdispl_;
  vector<int> rdispl_shape_;
  Dtype* tmp1_;
  vector<int> tmp1_shape_;
  Dtype* tmp2_;
  vector<int> tmp2_shape_;
  float voxel_relsize_z_;
  vector<float> rot_from_;
  vector<float> rot_to_;
  vector<float> offset_from_;
  vector<float> offset_to_;
  vector<int> mirror_flag_;
};

}  // namespace caffe

#endif  // CREATE_DEFORMATION_LAYER_HPP_

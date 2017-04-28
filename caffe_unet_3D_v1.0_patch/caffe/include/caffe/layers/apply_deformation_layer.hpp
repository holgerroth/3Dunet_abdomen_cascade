#ifndef APPLY_DEFORMATION_LAYER_HPP_
#define APPLY_DEFORMATION_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

/**
 * @brief Deforms an image with a given deformation field
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class ApplyDeformationLayer : public Layer<Dtype> {
 public:
  explicit ApplyDeformationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ApplyDeformation"; }

  /// @brief image, and deformation field
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  /// @brief deformed image
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int n_spatial_axes_;
  int n_deform_comps_;
};


}  // namespace caffe

#endif  // APPLY_DEFORMATION_LAYER_HPP_

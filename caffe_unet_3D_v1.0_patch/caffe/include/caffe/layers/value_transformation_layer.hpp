#ifndef CAFFE_VALUE_TRANSFORMATION_LAYER_
#define CAFFE_VALUE_TRANSFORMATION_LAYER_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Channel-wise affine intensity value transformation
 *    @f$ y_c = \alpha_c ( x_c + \beta_c ) @f$
 */
template <typename Dtype>
class ValueTransformationLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ValueTransformationParameter
   *     value_augmentation_param, with ValueTransformationLayer options:
   *
   *   - offset (\b optional, vectorial (1|#channels), default 0) @f$\beta@f$.
   *          Add the given constant value(s) to the channels of the bottom
   *          blob. If one value is given, it will be added to all channels,
   *          otherwise the number of given values must equal the number of
   *          channels and the corresponding individual channel offset is
   *          applied.
   *   - scale (\b optional, vectorial (1|#channels), default 1) @f$\alpha@f$.
   *          Scale the channels of the bottom blob with the given scale
   *          factor(s). If one value is given it will be used for all
   *          channels, otherwise the number of given values must equal the
   *          number of channels and the corresponding individual channel
   *          scale factors are applied.
   *
   *   The scale and offset are given, first the offset is applied and the
   *   result scaled.
   */
  explicit ValueTransformationLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}


  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ValueTransformation"; }

 protected:

  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y_c = \alpha_c (x + \beta_c)
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Dummy function -- no gradients yet
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  std::vector<Dtype> _offset, _scale;
};

}  // namespace caffe

#endif  // CAFFE_VALUE_TRANSFORMATION_LAYER_

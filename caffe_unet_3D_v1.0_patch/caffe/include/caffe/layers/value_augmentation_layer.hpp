#ifndef VALUE_AUGMENTATION_LAYER_HPP_
#define VALUE_AUGMENTATION_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Data augmentation by mapping the values. The layer provides
 * a contrast increse/decrease by using a smooth monotoning mapping.
 */
template <typename Dtype>
class ValueAugmentationLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ValueAugmentationParameter value_augmentation_param,
   *     with ValueAugmentationLayer options:
   *
   *     - black_from, black_to range from which to uniformly sample the
   *                            value mapped to black (i.e. zero)
   *
   *     - white_from, white_to range from which to uniformly sample the
   *                            value mapped to white (i.e. one)
   *
   *     - slope_min, slope_max range from which to uniformly sample the slopes
   *
   *     - n_control_point_insertions determines the number of different slopes
   */
  explicit ValueAugmentationLayer(const LayerParameter& param)
    : NeuronLayer<Dtype>(param) {}

  void CreateLinearInterpExtrapMatrix(
    int n_in,  Dtype dx_in,
    int n_out, Dtype dx_out,
    int n_extrapol,
    Dtype* lin_mat);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ValueAugmentation"; }

  std::vector<Dtype> random_lut_controlpoints(
    Dtype black_from, Dtype black_to,
    Dtype white_from, Dtype white_to,
    Dtype slope_min,  Dtype slope_max,
    int n_control_point_insertions);

  std::vector<Dtype> dense_lut(const std::vector<Dtype>& lut_control_points);

  void apply_lut(const std::vector<Dtype>& lut,
                 const Dtype* in_data, Dtype* out_data, size_t count);

 protected:

  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = f(x),
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  /**
   * @brief Dummy function -- no gradients yet
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial x} = ...
   *      @f$ if propagate_down[0]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  Dtype  black_from_;
  Dtype  black_to_;
  Dtype  white_from_;
  Dtype  white_to_;
  Dtype  slope_min_;
  Dtype  slope_max_;
  int    n_control_points_;
  int    n_control_point_insertions_;
  int    lut_size_;
  Blob<Dtype> interpol_mat_;
};

} // namespace cafff

#endif // VALUE_AUGMENTATION_LAYER_HPP_

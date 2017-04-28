#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/value_transformation_layer.hpp"

namespace caffe {

template <typename Dtype>
void ValueTransformationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  const ValueTransformationParameter& param =
      this->layer_param_.value_transformation_param();
  const int nchannels = bottom[0]->shape(1);
  CHECK(param.offset().v_size() <= 1 || param.offset().v_size() == nchannels)
      << "Specify either one offset or as many as channels: " << nchannels;
  CHECK(param.scale().v_size() <= 1 || param.scale().v_size() == nchannels)
      << "Specify either one scale or as many as channels: " << nchannels;
  _offset.resize(nchannels);
  _scale.resize(nchannels);
  if (param.offset().v_size() == 1) {
    for (int ch = 0; ch < nchannels; ++ch) _offset[ch] = param.offset().v(0);
  }
  else if (param.offset().v_size() == nchannels) {
    for (int ch = 0; ch < nchannels; ++ch) _offset[ch] = param.offset().v(ch);
  }
  else {
    for (int ch = 0; ch < nchannels; ++ch) _offset[ch] = 0.0f;
  }

  if (param.scale().v_size() == 1) {
    for (int ch = 0; ch < nchannels; ++ch) _scale[ch] = param.scale().v(0);
  }
  else if (param.scale().v_size() == nchannels) {
    for (int ch = 0; ch < nchannels; ++ch) _scale[ch] = param.scale().v(ch);
  }
  else {
    for (int ch = 0; ch < nchannels; ++ch) _scale[ch] = 1.0f;
  }
}

template <typename Dtype>
void ValueTransformationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int nsamples  = bottom[0]->shape(0);
  const int nchannels = bottom[0]->shape(1);
  const int count     = bottom[0]->count() / (nsamples * nchannels);

  caffe_copy(bottom[0]->count(), bottom_data, top_data);

  for (int num = 0; num < nsamples; ++num) {
    for (int ch = 0; ch < nchannels; ++ch) {
      if (_offset[ch] != Dtype(0)) {
        caffe_add_scalar(
            count, _offset[ch], top_data + (num * nchannels + ch) * count);
      }
      if (_scale[ch] != Dtype(1)) {
        caffe_scal(
            count, _scale[ch], top_data + (num * nchannels + ch) * count);
      }
    }
  }
}

template <typename Dtype>
void ValueTransformationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int nsamples  = bottom[0]->shape(0);
    const int nchannels = bottom[0]->shape(1);
    const int count     = bottom[0]->count() / (nsamples * nchannels);

    for (int num = 0; num < nsamples; ++num) {
      for (int ch = 0; ch < nchannels; ++ch) {
        // d/dx alpha_c * (x + beta_c) = alpha_c
        caffe_set(count, _scale[ch],
                  bottom_diff + (num * nchannels + ch) * count);
      }
    }
    caffe_mul(bottom[0]->count(), top_diff, bottom_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ValueTransformationLayer);
#endif

INSTANTIATE_CLASS(ValueTransformationLayer);
REGISTER_LAYER_CLASS(ValueTransformation);

}  // namespace caffe

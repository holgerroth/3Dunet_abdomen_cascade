#include <vector>

#include "caffe/layers/value_transformation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ValueTransformationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int nsamples  = bottom[0]->shape(0);
  const int nchannels = bottom[0]->shape(1);
  const int count     = bottom[0]->count() / (nsamples * nchannels);

  caffe_copy(bottom[0]->count(), bottom_data, top_data);

  for (int num = 0; num < nsamples; ++num) {
    for (int ch = 0; ch < nchannels; ++ch) {
      if (_offset[ch] != Dtype(0)) {
        caffe_gpu_add_scalar(
            count, _offset[ch], top_data + (num * nchannels + ch) * count);
      }
      if (_scale[ch] != Dtype(1)) {
        caffe_gpu_scal(
            count, _scale[ch], top_data + (num * nchannels + ch) * count);
      }
    }
  }
}

template <typename Dtype>
void ValueTransformationLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int nsamples  = bottom[0]->shape(0);
    const int nchannels = bottom[0]->shape(1);
    const int count     = bottom[0]->count() / (nsamples * nchannels);

    for (int num = 0; num < nsamples; ++num) {
      for (int ch = 0; ch < nchannels; ++ch) {
        // d/dx alpha_c * (x + beta_c) = alpha_c
        caffe_gpu_set(count, _scale[ch],
                      bottom_diff + (num * nchannels + ch) * count);
      }
    }
    caffe_gpu_mul(bottom[0]->count(), top_diff, bottom_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ValueTransformationLayer);

}

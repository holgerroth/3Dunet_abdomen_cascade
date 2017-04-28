#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensorDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorDesc<Dtype>(&top_desc_);
  vector<int> kernel_shape(this->kernel_shape_.count());
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  for (int d = 0; d < kernel_shape.size(); d++) {
      kernel_shape[d] = kernel_shape_data[d];
  }
  vector<int> pad_shape(this->pad_.count());
  int* pad_shape_data = this->pad_.mutable_cpu_data();
  for (int d = 0; d < pad_shape.size(); d++) {
    pad_shape[d] = pad_shape_data[d];
  }
  vector<int> stride_shape(this->stride_.count());
  int* stride_shape_data = this->stride_.mutable_cpu_data();
  for (int d = 0; d < stride_shape.size(); d++) {
    stride_shape[d] = stride_shape_data[d];
  }
  cudnn::createNdPoolingDesc<Dtype>(&pooling_desc_,
      this->layer_param_.pooling_param().pool(), &mode_,
      kernel_shape,pad_shape,stride_shape);
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::Reshape(bottom, top);
  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, bottom[0]->shape());
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, top[0]->shape());
}

template <typename Dtype>
CuDNNPoolingLayer<Dtype>::~CuDNNPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyPoolingDescriptor(pooling_desc_);
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS(CuDNNPoolingLayer);

}   // namespace caffe
#endif

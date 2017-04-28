#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    printf("label_value = %g\n",label_value);
    printf("ignore_label_ = %g\n",ignore_label_);
    printf("has_ignore_label_ = %d\n",has_ignore_label_);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void WeightedSoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, const Dtype* weights,
          Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    /*if (label_value>0 && label_value != ignore_label_){
	    printf("label_value = %d\n",label_value);
	    //printf("ignore_label_ = %d\n",ignore_label_);
	    //printf("has_ignore_label_ = %d\n",has_ignore_label_);
    }*/
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      Dtype weight = weights[n * spatial_dim + s];
      //printf("weight = %g\n",weight);
      loss[index] = weight *
          -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                   Dtype(FLT_MIN)));
      counts[index] = weight;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  if( bottom.size() == 2) {
    // original version with equally weighted pixels
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  } else {
    // version with pixel-wise loss weights using a third input blob
    WeightedSoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label,
        bottom[2]->gpu_data(), loss_data,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  }
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.

  if ( (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) || bottom.size() == 3) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
    if( valid_count == 0 ) {
      LOG(INFO) << this->type()
		<< " warning (Forward_gpu): sum of pixel wise loss weights is zero!";
    }
  }
  if ( valid_count == 0) {
      top[0]->mutable_cpu_data()[0] = 0.;
  } else {
      top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                          valid_count);
  }

  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void WeightedSoftmaxLossBackwardGPU(const int nthreads,
          const Dtype* top, const Dtype* label, const Dtype* weights,
          Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      Dtype weight = weights[n * spatial_dim + s];
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] *= weight;
      }
      counts[index] = weight;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();

   if( bottom.size() == 2) {
      // original version with equally weighted pixels
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,
          counts);
    } else {
       WeightedSoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label,
          bottom[2]->gpu_data(), bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,
          counts);
    }
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if ( (normalization_ == LossParameter_NormalizationMode_VALID &&
          has_ignore_label_) || (bottom.size() == 3) ) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    Dtype loss_weight = 0;
    if( valid_count == 0) {
        LOG(INFO) << this->type()
                  << " warning (Backward_gpu): sum of pixel wise loss weights is zero!";
    } else {
        loss_weight = top[0]->cpu_diff()[0] /
            get_normalizer(normalization_, valid_count);
    }
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe

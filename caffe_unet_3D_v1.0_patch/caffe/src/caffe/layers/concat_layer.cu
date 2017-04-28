#include <string>
#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}


// Copy (one line per thread) from one array to another, with arbitrary
// strides in all other dimensions
template <typename Dtype>
__global__ void copy_subarray_4D(
    const int nthreads,    const int shape1, const int shape2, const int shape3,
    const int src_stride0, const int src_stride1, const int src_stride2,
    const int dest_stride0, const int dest_stride1, const int dest_stride2,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i0 = index / (shape1 * shape2);
    int r0 = index % (shape1 * shape2);
    int i1 = r0 / shape2;
    int i2 = r0 % shape2;
    int src_start  = i0 * src_stride0  + i1 * src_stride1  + i2 * src_stride2;
    int dest_start = i0 * dest_stride0 + i1 * dest_stride1 + i2 * dest_stride2;
    for (int i3 = 0; i3 < shape3; ++i3) {
        dest[dest_start + i3] = src[src_start + i3];
    }
  }
}

// Copy (one line per thread) from one array to another, with arbitrary
// strides in all other dimensions
template <typename Dtype>
__global__ void copy_subarray_5D(
    const int nthreads,    const int shape1, const int shape2, const int shape3, const int shape4,
    const int src_stride0, const int src_stride1, const int src_stride2, const int src_stride3,
    const int dest_stride0, const int dest_stride1, const int dest_stride2, const int dest_stride3,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i0 = index / (shape1 * shape2 * shape3);
    int r0 = index % (shape1 * shape2 * shape3);
    int i1 = r0 / (shape2 * shape3);
    int r1 = r0 % (shape2 * shape3);
    int i2 = r1 / shape3;
    int i3 = r1 % shape3;
    int src_start  = i0 * src_stride0  + i1 * src_stride1  + i2 * src_stride2  + i3 * src_stride3;
    int dest_start = i0 * dest_stride0 + i1 * dest_stride1 + i2 * dest_stride2 + i3 * dest_stride3;
    for (int i4 = 0; i4 < shape4; ++i4) {
        dest[dest_start + i4] = src[src_start + i4];
    }
  }
}


template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  if( needs_cropping_ == false) {
    // original code for concatenation without cropping
  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = true;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int nthreads = bottom_concat_size * num_concats_;
    Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
        top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
    offset_concat_axis += bottom_concat_axis;
    }
  } else {
    // concatenation with cropping of input blobs
    int offset_concat_axis = 0;
    const int num_axes = bottom[0]->num_axes();

    for (int i = 0; i < bottom.size(); ++i) {
      // compute offsets and shape of copy region
      vector<int> bottom_offset(num_axes);
      for (int j = 0; j < num_axes; ++j) {
        bottom_offset[j] = (bottom[i]->shape(j) - top[0]->shape(j))/2;
      }
      bottom_offset[concat_axis_] = 0;
      vector<int> copy_shape(num_axes);
      copy_shape = top[0]->shape();
      copy_shape[concat_axis_] = bottom[i]->shape(concat_axis_);
      vector<int> top_offset(num_axes,0);
      top_offset[concat_axis_] = offset_concat_axis;

      if (num_axes == 4 && concat_axis_ == 1) {
        // fast code for blobs with 4 axes (2 spatial dims))
        // one thread per line
        const int nthreads        = copy_shape[0] * copy_shape[1] * copy_shape[2];
        const int bottom_stride0  = bottom[i]->shape(1) * bottom[i]->shape(2) * bottom[i]->shape(3);
        const int bottom_stride1  = bottom[i]->shape(2) * bottom[i]->shape(3);
        const int bottom_stride2  = bottom[i]->shape(3);
        const int top_stride0     = top[0]->shape(1) * top[0]->shape(2) * top[0]->shape(3);
        const int top_stride1     = top[0]->shape(2) * top[0]->shape(3);
        const int top_stride2     = top[0]->shape(3);
        const Dtype* bottom_start = bottom[i]->gpu_data() + bottom[i]->offset( bottom_offset);
        Dtype*       top_start    = top[0]->mutable_gpu_data() + top[0]->offset( top_offset);
        copy_subarray_4D<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                nthreads,  copy_shape[1], copy_shape[2], copy_shape[3],
                bottom_stride0,  bottom_stride1,  bottom_stride2,
                top_stride0,  top_stride1,  top_stride2,
                bottom_start, top_start);
      } else {
	if (num_axes == 5 && concat_axis_ == 1) {
        // fast code for blobs with 5 axes (3 spatial dims))
        // one thread per line
        const int nthreads        = copy_shape[0] * copy_shape[1] * copy_shape[2] * copy_shape[3];
        const int bottom_stride0  = bottom[i]->shape(1) * bottom[i]->shape(2) * bottom[i]->shape(3)
	                            * bottom[i]->shape(4);
        const int bottom_stride1  = bottom[i]->shape(2) * bottom[i]->shape(3) * bottom[i]->shape(4);
        const int bottom_stride2  = bottom[i]->shape(3) * bottom[i]->shape(4);
	const int bottom_stride3  = bottom[i]->shape(4);
        const int top_stride0     = top[0]->shape(1) * top[0]->shape(2) * top[0]->shape(3) * top[0]->shape(4);
        const int top_stride1     = top[0]->shape(2) * top[0]->shape(3) * top[0]->shape(4);
        const int top_stride2     = top[0]->shape(3) * top[0]->shape(4);
	const int top_stride3     = top[0]->shape(4);
        const Dtype* bottom_start = bottom[i]->gpu_data() + bottom[i]->offset( bottom_offset);
        Dtype*       top_start    = top[0]->mutable_gpu_data() + top[0]->offset( top_offset);
        copy_subarray_5D<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
	        nthreads,  copy_shape[1], copy_shape[2], copy_shape[3], copy_shape[4],
                bottom_stride0,  bottom_stride1,  bottom_stride2, bottom_stride3,
                top_stride0,  top_stride1,  top_stride2, top_stride3,
                bottom_start, top_start);
	} else {
        // slow code for blobs with up to 10 axes
        caffe_copy_subarray( bottom[i]->gpu_data(), bottom[i]->shape(),
                             top[0]->mutable_gpu_data(), top[0]->shape(),
                             bottom_offset, copy_shape, top_offset);
        }
      }
      offset_concat_axis += bottom[i]->shape(concat_axis_);
    }
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() == 1) { return; }
  if( needs_cropping_ == false) {
    // original code for concatenation without cropping
  const Dtype* top_diff = top[0]->gpu_diff();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = false;
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
      const int nthreads = bottom_concat_size * num_concats_;
      Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, top_diff, kForward, num_concats_, concat_input_size_,
          top_concat_axis, bottom_concat_axis, offset_concat_axis, bottom_diff);
    }
    offset_concat_axis += bottom_concat_axis;
    }
  } else {
    // concatenation with cropping
    int offset_concat_axis = 0;
    const int num_axes = bottom[0]->num_axes();
    for (int i = 0; i < bottom.size(); ++i) {
      // initialize diff blobs to zero (beause gradients are
      // only available for cropped region)
      caffe_gpu_set(bottom[i]->count(), static_cast<Dtype>(0),
                    bottom[i]->mutable_gpu_diff());
      // compute offsets and shape of copy region
      vector<int> bottom_offset(num_axes);
      for (int j = 0; j < num_axes; ++j) {
        bottom_offset[j] = (bottom[i]->shape(j) - top[0]->shape(j))/2;
      }
      bottom_offset[concat_axis_] = 0;
      vector<int> copy_shape(num_axes);
      copy_shape = top[0]->shape();
      copy_shape[concat_axis_] = bottom[i]->shape(concat_axis_);
      vector<int> top_offset(num_axes,0);
      top_offset[concat_axis_] = offset_concat_axis;
      if (num_axes == 4 && concat_axis_ == 1) {
        // fast code for blobs with 4 axes (2 spatial dims))
        // one thread per line
        const int nthreads        = copy_shape[0] * copy_shape[1] * copy_shape[2];
        const int bottom_stride0  = bottom[i]->shape(1) * bottom[i]->shape(2) * bottom[i]->shape(3);
        const int bottom_stride1  = bottom[i]->shape(2) * bottom[i]->shape(3);
        const int bottom_stride2  = bottom[i]->shape(3);
        const int top_stride0     = top[0]->shape(1) * top[0]->shape(2) * top[0]->shape(3);
        const int top_stride1     = top[0]->shape(2) * top[0]->shape(3);
        const int top_stride2     = top[0]->shape(3);
        Dtype*       bottom_start = bottom[i]->mutable_gpu_diff() + bottom[i]->offset( bottom_offset);
        const Dtype* top_start    = top[0]->gpu_diff() + top[0]->offset( top_offset);
        copy_subarray_4D<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                nthreads,  copy_shape[1], copy_shape[2], copy_shape[3],
                top_stride0,  top_stride1,  top_stride2,
                bottom_stride0,  bottom_stride1,  bottom_stride2,
                top_start, bottom_start);
      } else {
	if (num_axes == 5 && concat_axis_ == 1) {
        // fast code for blobs with 5 axes (3 spatial dims))
        // one thread per line
        const int nthreads        = copy_shape[0] * copy_shape[1] * copy_shape[2] * copy_shape[3];
        const int bottom_stride0  = bottom[i]->shape(1) * bottom[i]->shape(2) * bottom[i]->shape(3)
	                            * bottom[i]->shape(4);
        const int bottom_stride1  = bottom[i]->shape(2) * bottom[i]->shape(3) * bottom[i]->shape(4);
        const int bottom_stride2  = bottom[i]->shape(3) * bottom[i]->shape(4);
	const int bottom_stride3  = bottom[i]->shape(4);
        const int top_stride0     = top[0]->shape(1) * top[0]->shape(2) * top[0]->shape(3) * top[0]->shape(4);
        const int top_stride1     = top[0]->shape(2) * top[0]->shape(3) * top[0]->shape(4);
        const int top_stride2     = top[0]->shape(3) * top[0]->shape(4);
	const int top_stride3     = top[0]->shape(4);
        Dtype*       bottom_start = bottom[i]->mutable_gpu_diff() + bottom[i]->offset( bottom_offset);
        const Dtype* top_start    = top[0]->gpu_diff() + top[0]->offset( top_offset);
        copy_subarray_5D<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
		nthreads,  copy_shape[1], copy_shape[2], copy_shape[3], copy_shape[4],
                top_stride0,  top_stride1,  top_stride2, top_stride3,
                bottom_stride0,  bottom_stride1,  bottom_stride2, bottom_stride3,
                top_start, bottom_start);
	} else {
        // slow code for blobs with up to 10 axes
        caffe_copy_subarray( top[0]->gpu_diff(), top[0]->shape(),
                             bottom[i]->mutable_gpu_diff(), bottom[i]->shape(),
                             top_offset, copy_shape, bottom_offset);
        }
      }
      offset_concat_axis += bottom[i]->shape(concat_axis_);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);

}  // namespace caffe

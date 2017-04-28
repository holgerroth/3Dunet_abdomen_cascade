#include <string>
#include <sstream>
#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/vector_helper.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";
}

template <typename Dtype>
void ConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  if (concat_param.has_concat_dim()) {
    concat_axis_ = static_cast<int>(concat_param.concat_dim());
    // Don't allow negative indexing for concat_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(concat_axis_, 0) << "casting concat_dim from uint32 to int32 "
        << "produced negative result; concat_dim must satisfy "
        << "0 <= concat_dim < " << kMaxBlobAxes;
    CHECK_LT(concat_axis_, num_axes) << "concat_dim out of range.";
  } else {
    concat_axis_ = bottom[0]->CanonicalAxisIndex(concat_param.axis());
  }
  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
  num_concats_ = bottom[0]->count(0, concat_axis_);
  concat_input_size_ = bottom[0]->count(concat_axis_ + 1);
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) { continue; }
      CHECK_LE(top_shape[j], bottom[i]->shape(j))
          << "All inputs must have the same or greater shape like the first blob, except at concat_axis.";
    }
    top_shape[concat_axis_] += bottom[i]->shape(concat_axis_);
  }
  bool shape_changed = (top[0]->shape() != top_shape);
  needs_cropping_ = false;
  top[0]->Reshape(top_shape);
  if (bottom.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
    return;
  }

  // Olaf: compute the borderwidth's for the input blobs
  // and check whether cropping is needed
  CHECK_LT( num_axes, 10) << "only 10 axes are supported";
  for (int i = 0; i < bottom.size(); ++i) {
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) { continue; }
      int width_difference = bottom[i]->shape(j) - top_shape[j];
      if (width_difference != 0)
      {
        needs_cropping_ = true;

        int borderwidth = width_difference/2;
        CHECK_EQ(borderwidth*2, width_difference) <<
            "width difference must be even! input blob " << i << " axis " << j << " has width " << bottom[i]->shape(j) << " and output blob has width " << top_shape[j] << ". The difference " << width_difference << " is not even.";
      }
    }
  }

  // Olaf: compute the borderwidth's for the input blobs
  // and check whether cropping is needed
  CHECK_LT( num_axes, 10) << "only 10 axes are supported";
  for (int i = 0; i < bottom.size(); ++i) {
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) { continue; }
      int width_difference = bottom[i]->shape(j) - top_shape[j];
      if (width_difference != 0)
      {
        needs_cropping_ = true;

        int borderwidth = width_difference/2;
        CHECK_EQ(borderwidth*2, width_difference) <<
            "width difference must be even! input blob " << i << " axis " << j << " has width " << bottom[i]->shape(j) << " and output blob has width " << top_shape[j] << ". The difference " << width_difference << " is not even.";
      }
    }
    if (shape_changed && needs_cropping_) {
      vector<int> outShape( top_shape);
      outShape[concat_axis_] = bottom[i]->shape(concat_axis_);
      LOG(INFO) << "bottom blob " << i << " " << toString(bottom[i]->shape())
                << " will be cropped to " << toString(outShape);
    }
  }


}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  if( needs_cropping_ == false) {
    // original code for concatenation without cropping
    Dtype* top_data = top[0]->mutable_cpu_data();
    int offset_concat_axis = 0;
    const int top_concat_axis = top[0]->shape(concat_axis_);
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
      for (int n = 0; n < num_concats_; ++n) {
	caffe_copy(bottom_concat_axis * concat_input_size_,
		   bottom_data + n * bottom_concat_axis * concat_input_size_,
		   top_data + (n * top_concat_axis + offset_concat_axis)
		   * concat_input_size_);
      }
      offset_concat_axis += bottom_concat_axis;
    }
  } else {
    // concatenation with cropping of input blobs
    int offset_concat_axis = 0;
    const int num_axes = bottom[0]->num_axes();
    for (int i = 0; i < bottom.size(); ++i) {
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
      caffe_copy_subarray( bottom[i]->cpu_data(), bottom[i]->shape(),
			   top[0]->mutable_cpu_data(), top[0]->shape(),
			   bottom_offset, copy_shape, top_offset);
      offset_concat_axis += bottom[i]->shape(concat_axis_);
    }
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() == 1) { return; }
  if( needs_cropping_ == false) {
    // original code for concatenation without cropping
    const Dtype* top_diff = top[0]->cpu_diff();
    int offset_concat_axis = 0;
    const int top_concat_axis = top[0]->shape(concat_axis_);
    for (int i = 0; i < bottom.size(); ++i) {
      const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
      if (propagate_down[i]) {
	Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
	for (int n = 0; n < num_concats_; ++n) {
	  caffe_copy(bottom_concat_axis * concat_input_size_, top_diff +
		     (n * top_concat_axis + offset_concat_axis) * concat_input_size_,
		     bottom_diff + n * bottom_concat_axis * concat_input_size_);
	}
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
      caffe_set(bottom[i]->count(), static_cast<Dtype>(0),
                bottom[i]->mutable_cpu_diff());
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
      caffe_copy_subarray( top[0]->cpu_diff(), top[0]->shape(),
                           bottom[i]->mutable_cpu_diff(), bottom[i]->shape(),
                           top_offset, copy_shape, bottom_offset);
      offset_concat_axis += bottom[i]->shape(concat_axis_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConcatLayer);
#endif

INSTANTIATE_CLASS(ConcatLayer);
REGISTER_LAYER_CLASS(Concat);

}  // namespace caffe

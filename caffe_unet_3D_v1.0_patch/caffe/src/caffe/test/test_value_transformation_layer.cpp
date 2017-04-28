#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/value_transformation_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

  template <typename TypeParam>
  class ValueTransformationLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

  protected:
    ValueTransformationLayerTest()
            : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
              blob_top_(new Blob<Dtype>()) {
      Caffe::set_random_seed(1527);
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);

      std::vector<int> shape_3d(5);
      shape_3d[0] = 2; // nsamples
      shape_3d[1] = 3; // nchannels
      shape_3d[2] = 4; // zsize (d)
      shape_3d[3] = 5; // ysize (h)
      shape_3d[4] = 6; // xsize (w)
      blob_bottom_3d_ = new Blob<Dtype>(shape_3d);
      blob_top_3d_ = new Blob<Dtype>(shape_3d);
      blob_bottom_vec_3d_.push_back(blob_bottom_3d_);
      blob_top_vec_3d_.push_back(blob_top_3d_);
    }
    virtual ~ValueTransformationLayerTest() {
      delete blob_bottom_; delete blob_top_;
      delete blob_bottom_3d_; delete blob_top_3d_;
    }

    void TestForward(std::vector<Dtype> const &scale,
                     std::vector<Dtype> const &offset) {
      LayerParameter layer_param;
      ValueTransformationParameter *value_transformation_param =
          layer_param.mutable_value_transformation_param();
      int nsamples = this->blob_bottom_->shape(0);
      int nchannels = this->blob_bottom_->shape(1);
      int count = this->blob_bottom_->count() / (nsamples * nchannels);
      for (size_t ch = 0; ch < offset.size(); ++ch) {
        value_transformation_param->mutable_offset()->add_v(offset[ch]);
      }
      for (size_t ch = 0; ch < scale.size(); ++ch) {
        value_transformation_param->mutable_scale()->add_v(scale[ch]);
      }

      std::vector<Dtype> offs(nchannels), sc(nchannels);
      for (size_t ch = 0; ch < offset.size(); ++ch) {
        offs[ch] = offset[ch];
      }
      for (size_t ch = offset.size(); ch < nchannels; ++ch) {
        offs[ch] = offset[offset.size() - 1];
      }
      for (size_t ch = 0; ch < scale.size(); ++ch) {
        sc[ch] = scale[ch];
      }
      for (size_t ch = scale.size(); ch < nchannels; ++ch) {
        sc[ch] = scale[scale.size() - 1];
      }

      ValueTransformationLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      // Now, check values
      const Dtype* bottom_data = this->blob_bottom_->cpu_data();
      const Dtype* top_data = this->blob_top_->cpu_data();
      const Dtype min_precision = 1e-5;
      for (int num = 0; num < nsamples; ++num) {
        for (int ch = 0; ch < nchannels; ++ch) {
          for (int i = 0; i < count; ++i) {
            Dtype expected_value = sc[ch] *
                (bottom_data[(num * nchannels + ch) * count + i] + offs[ch]);
            Dtype precision = std::max(
                Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
            EXPECT_NEAR(
                expected_value, top_data[(num * nchannels + ch) * count + i],
                precision);
          }
        }
      }
    }

    void TestForward3d(std::vector<Dtype> const &scale,
                       std::vector<Dtype> const &offset) {
      LayerParameter layer_param;
      ValueTransformationParameter *value_transformation_param =
          layer_param.mutable_value_transformation_param();
      int nsamples = this->blob_bottom_3d_->shape(0);
      int nchannels = this->blob_bottom_3d_->shape(1);
      int count = this->blob_bottom_3d_->count() / (nsamples * nchannels);
      for (size_t ch = 0; ch < offset.size(); ++ch) {
        value_transformation_param->mutable_offset()->add_v(offset[ch]);
      }
      for (size_t ch = 0; ch < scale.size(); ++ch) {
        value_transformation_param->mutable_scale()->add_v(scale[ch]);
      }

      std::vector<Dtype> offs(nchannels), sc(nchannels);
      for (size_t ch = 0; ch < offset.size(); ++ch) {
        offs[ch] = offset[ch];
      }
      for (size_t ch = offset.size(); ch < nchannels; ++ch) {
        offs[ch] = offset[offset.size() - 1];
      }
      for (size_t ch = 0; ch < scale.size(); ++ch) {
        sc[ch] = scale[ch];
      }
      for (size_t ch = scale.size(); ch < nchannels; ++ch) {
        sc[ch] = scale[scale.size() - 1];
      }

      ValueTransformationLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_3d_, this->blob_top_vec_3d_);
      layer.Forward(this->blob_bottom_vec_3d_, this->blob_top_vec_3d_);
      // Now, check values
      const Dtype* bottom_data = this->blob_bottom_3d_->cpu_data();
      const Dtype* top_data = this->blob_top_3d_->cpu_data();
      const Dtype min_precision = 1e-5;
      for (int num = 0; num < nsamples; ++num) {
        for (int ch = 0; ch < nchannels; ++ch) {
          for (int i = 0; i < count; ++i) {
            Dtype expected_value = sc[ch] *
                (bottom_data[(num * nchannels + ch) * count + i] + offs[ch]);
            Dtype precision = std::max(
                Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
            EXPECT_NEAR(
                expected_value, top_data[(num * nchannels + ch) * count + i],
                precision);
          }
        }
      }
    }

    void TestBackward(std::vector<Dtype> const &scale,
                      std::vector<Dtype> const &offset) {
      LayerParameter layer_param;
      ValueTransformationParameter *value_transformation_param =
          layer_param.mutable_value_transformation_param();
      for (size_t ch = 0; ch < offset.size(); ++ch) {
        value_transformation_param->mutable_offset()->add_v(offset[ch]);
      }
      for (size_t ch = 0; ch < scale.size(); ++ch) {
        value_transformation_param->mutable_scale()->add_v(scale[ch]);
      }
      ValueTransformationLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-3, 1e-2, 1527, 0., 0.01);
      checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
                                   this->blob_top_vec_);
    }

    void TestBackward3d(std::vector<Dtype> const &scale,
                        std::vector<Dtype> const &offset) {
      LayerParameter layer_param;
      ValueTransformationParameter *value_transformation_param =
          layer_param.mutable_value_transformation_param();
      for (size_t ch = 0; ch < offset.size(); ++ch) {
        value_transformation_param->mutable_offset()->add_v(offset[ch]);
      }
      for (size_t ch = 0; ch < scale.size(); ++ch) {
        value_transformation_param->mutable_scale()->add_v(scale[ch]);
      }
      ValueTransformationLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-3, 1e-2, 1527, 0., 0.01);
      checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_3d_,
                                   this->blob_top_vec_3d_);
    }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;

    Blob<Dtype>* blob_bottom_3d_;
    Blob<Dtype>* blob_top_3d_;
    vector<Blob<Dtype>*> blob_bottom_vec_3d_;
    vector<Blob<Dtype>*> blob_top_vec_3d_;
  };

  TYPED_TEST_CASE(ValueTransformationLayerTest, TestDtypesAndDevices);

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformation) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(3), offset(3);
    scale[0] = 0.62;
    scale[1] = 1.35;
    scale[2] = -1.5;
    offset[0] = -0.25;
    offset[1] = 1.8;
    offset[2] = -25.05;
    this->TestForward(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformation3d) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(3), offset(3);
    scale[0] = 0.62;
    scale[1] = 1.35;
    scale[2] = -1.5;
    offset[0] = -0.25;
    offset[1] = 1.8;
    offset[2] = -25.05;
    this->TestForward3d(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformationGradient) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(3), offset(3);
    scale[0] = 0.62;
    scale[1] = 1.35;
    scale[2] = -1.5;
    offset[0] = -0.25;
    offset[1] = 1.8;
    offset[2] = -25.05;
    this->TestBackward(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformationGradient3d) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(3), offset(3);
    scale[0] = 0.62;
    scale[1] = 1.35;
    scale[2] = -1.5;
    offset[0] = -0.25;
    offset[1] = 1.8;
    offset[2] = -25.05;
    this->TestBackward3d(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformationOneOffset) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(3), offset(1);
    scale[0] = 0.62;
    scale[1] = 1.35;
    scale[2] = -1.5;
    offset[0] = -0.25;
    this->TestForward(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformationOneOffset3d) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(3), offset(1);
    scale[0] = 0.62;
    scale[1] = 1.35;
    scale[2] = -1.5;
    offset[0] = -0.25;
    this->TestForward3d(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformationOneOffsetGradient) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(3), offset(1);
    scale[0] = 0.62;
    scale[1] = 1.35;
    scale[2] = -1.5;
    offset[0] = -0.25;
    this->TestBackward(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformationOneOffsetGradient3d) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(3), offset(1);
    scale[0] = 0.62;
    scale[1] = 1.35;
    scale[2] = -1.5;
    offset[0] = -0.25;
    this->TestBackward3d(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformationOneScale) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(1), offset(3);
    scale[0] = 0.62;
    offset[0] = -0.25;
    offset[1] = 1.8;
    offset[2] = -25.05;
    this->TestForward(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformationOneScale3d) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(1), offset(3);
    scale[0] = 0.62;
    offset[0] = -0.25;
    offset[1] = 1.8;
    offset[2] = -25.05;
    this->TestForward3d(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformationOneScaleGradient) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(1), offset(3);
    scale[0] = 0.62;
    offset[0] = -0.25;
    offset[1] = 1.8;
    offset[2] = -25.05;
    this->TestBackward(scale, offset);
  }

  TYPED_TEST(ValueTransformationLayerTest, TestValueTransformationOneScaleGradient3d) {
    typedef typename TypeParam::Dtype Dtype;
    std::vector<Dtype> scale(1), offset(3);
    scale[0] = 0.62;
    offset[0] = -0.25;
    offset[1] = 1.8;
    offset[2] = -25.05;
    this->TestBackward3d(scale, offset);
  }

}  // namespace caffe

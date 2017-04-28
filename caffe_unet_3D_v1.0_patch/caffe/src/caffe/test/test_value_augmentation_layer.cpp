#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/layers/value_augmentation_layer.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/vector_helper.hpp"

namespace caffe {


template <typename TypeParam>
class ValueAugmentationLayerTest : public ::testing::Test  {

 protected:
  ValueAugmentationLayerTest()
    : blob_bottom_(new Blob<TypeParam>(2, 3, 4, 5)),
      blob_top_(new Blob<TypeParam>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ValueAugmentationLayerTest() { delete blob_bottom_; delete blob_top_; }

  Blob<TypeParam>* const blob_bottom_;
  Blob<TypeParam>* const blob_top_;
  vector<Blob<TypeParam>*> blob_bottom_vec_;
  vector<Blob<TypeParam>*> blob_top_vec_;
};

TYPED_TEST_CASE(ValueAugmentationLayerTest, TestDtypes);

TYPED_TEST(ValueAugmentationLayerTest, TestLinearInterpExtrapMatrix) {
  LayerParameter layer_param;
  ValueAugmentationLayer<TypeParam> layer(layer_param);

  int n_in         = 5;
  TypeParam dx_in  = 0.25;
  int n_out        = 101;
  TypeParam dx_out = 0.01;
  int n_extrapol   = 50;

  TypeParam* lin_mat = new TypeParam[n_in * (n_out + 2 * n_extrapol)];
  layer.CreateLinearInterpExtrapMatrix(n_in, dx_in, n_out, dx_out,
                                       n_extrapol, lin_mat);
  std::vector<TypeParam> lut_cp(5);
  for (int i = 0; i < lut_cp.size(); ++i) {
    lut_cp[i] = 0.25 * i;
  }

  std::vector<TypeParam> lut(201);
  caffe_cpu_gemv<TypeParam>( CblasNoTrans,
                             201, 5,
                             TypeParam(1), lin_mat, lut_cp.data(),
                             TypeParam(0), lut.data());

  for (int i = 0; i < lut.size(); ++i) {
    EXPECT_NEAR( lut[i], 0.01 * (i - n_extrapol), 1e-6);
  }
}

TYPED_TEST(ValueAugmentationLayerTest, TestRandomControlPoints) {
  LayerParameter layer_param;
  ValueAugmentationLayer<TypeParam> layer(layer_param);

  std::vector<TypeParam> lut_cp = layer.random_lut_controlpoints( 0,0,
                                                                  1,1,
                                                                  1,1, 2);
  EXPECT_EQ( 5, lut_cp.size());

  for (int i = 0; i < lut_cp.size(); ++i) {
    EXPECT_NEAR( lut_cp[i], 0.25 * i, 1e-6);
  }

  for (int iter = 0; iter < 10; ++iter) {
    lut_cp = layer.random_lut_controlpoints( 0, 0, 1, 1, 0.5, 2, 2);
    EXPECT_NEAR( lut_cp[0], 0, 1e-6);
    EXPECT_NEAR( lut_cp[4], 1, 1e-6);

    // test slopes
    for (int i = 0; i < lut_cp.size() - 1; ++i) {
      EXPECT_GE( (lut_cp[i+1] - lut_cp[i]) / 0.25, 0.5);
      EXPECT_LE( (lut_cp[i+1] - lut_cp[i]) / 0.25, 2);
    }
  }
}

TYPED_TEST(ValueAugmentationLayerTest, TestInterpolationAndSmoothing) {
  LayerParameter layer_param;
  layer_param.mutable_value_augmentation_param()->set_lut_size(101);
  ValueAugmentationLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  std::vector<TypeParam> lut_cp(5);
  for (int i = 0; i < lut_cp.size(); ++i) {
    lut_cp[i] = 0.25 * i;
  }

  std::vector<TypeParam> lut = layer.dense_lut( lut_cp);

  EXPECT_EQ( 101, lut.size());

  for (int i = 0; i < lut.size(); ++i) {
    EXPECT_NEAR( lut[i], 0.01 * i, 1e-6);
  }
}

TYPED_TEST(ValueAugmentationLayerTest, TestIdentityMapping) {
  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  UniformFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);

  LayerParameter layer_param;
  layer_param.mutable_value_augmentation_param()->set_slope_min(1);
  layer_param.mutable_value_augmentation_param()->set_slope_max(1);
  ValueAugmentationLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(top_data[i], bottom_data[i], 1e-6);
  }
}

TYPED_TEST(ValueAugmentationLayerTest, TestScale) {
  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  UniformFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);

  LayerParameter layer_param;
  layer_param.mutable_value_augmentation_param()->set_white_from(2);
  layer_param.mutable_value_augmentation_param()->set_white_to(2);
  layer_param.mutable_value_augmentation_param()->set_slope_min(2);
  layer_param.mutable_value_augmentation_param()->set_slope_max(2);
  ValueAugmentationLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 2 * bottom_data[i], 1e-4);
  }
}

#if 0
TYPED_TEST(ValueAugmentationLayerTest, TestSCurve) {
  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  UniformFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);

  LayerParameter layer_param;
  layer_param.mutable_value_augmentation_param()->set_s_curve_contrast_max(10);
  ValueAugmentationLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_data()[i] = 42;
  }

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();

  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE( top_data[i], 0);
    EXPECT_LE( top_data[i], 1);
    EXPECT_NE( top_data[i], bottom_data[i]);
  }
}

TYPED_TEST(ValueAugmentationLayerTest, TestGamma) {
  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  UniformFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);

  LayerParameter layer_param;
  layer_param.mutable_value_augmentation_param()->set_gamma_max(10);
  ValueAugmentationLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_data()[i] = 42;
  }

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();

  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE( top_data[i], 0);
    EXPECT_LE( top_data[i], 1);
    EXPECT_NE( top_data[i], bottom_data[i]);
  }
}
#endif

}  // namespace caffe

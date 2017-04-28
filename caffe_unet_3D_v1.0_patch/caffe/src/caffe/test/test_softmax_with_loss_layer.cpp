#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxWithLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxWithLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_bottom_pixel_loss_weight_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
    // bottom Blobs for pixel wise loss weight
    blob_bottom_vec_3_.push_back(blob_bottom_data_);
    blob_bottom_vec_3_.push_back(blob_bottom_label_);
    // fill weights with positive gaussian distributed random numbers
    filler.Fill(this->blob_bottom_pixel_loss_weight_);
    for (int i = 0; i < blob_bottom_pixel_loss_weight_->count(); ++i) {
      blob_bottom_pixel_loss_weight_->mutable_cpu_data()[i] =
          abs(blob_bottom_pixel_loss_weight_->cpu_data()[i]);
    }
    blob_bottom_vec_3_.push_back(blob_bottom_pixel_loss_weight_);

  }
  virtual ~SoftmaxWithLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_pixel_loss_weight_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_3_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxWithLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  // First, compute the loss with all labels
  scoped_ptr<SoftmaxWithLossLayer<Dtype> > layer(
      new SoftmaxWithLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
  // Now, accumulate the loss, ignoring each label in {0, ..., 4} in turn.
  Dtype accum_loss = 0;
  for (int label = 0; label < 5; ++label) {
    layer_param.mutable_loss_param()->set_ignore_label(label);
    layer.reset(new SoftmaxWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    accum_loss += this->blob_top_loss_->cpu_data()[0];
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(4 * full_loss, accum_loss, 1e-4);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradientIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // labels are in {0, ..., 4}, so we'll ignore about a fifth of them
  layer_param.mutable_loss_param()->set_ignore_label(0);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestForwardPixelWiseLossWeight) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_3_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestForwardPixelWiseLossWeight2) {
  // simulate ignore label test using PixelWiseLossWeight
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  // First, compute the loss with all labels
  scoped_ptr<SoftmaxWithLossLayer<Dtype> > layer(
      new SoftmaxWithLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_3_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_3_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];

  Blob<Dtype> backup;
  backup.CopyFrom( *(this->blob_bottom_pixel_loss_weight_), false, true);

  // Now, accumulate the loss, ignoring each label in {0, ..., 4} in turn.
  Dtype accum_loss = 0;
  for (int label = 0; label < 5; ++label) {
    this->blob_bottom_pixel_loss_weight_->CopyFrom( backup);
    for (int i = 0; i < this->blob_bottom_pixel_loss_weight_->count(); ++i) {
      if( this->blob_bottom_label_->cpu_data()[i] == label) {
        this->blob_bottom_pixel_loss_weight_->mutable_cpu_data()[i] = 0;
      }
    }
    layer->SetUp(this->blob_bottom_vec_3_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_3_, this->blob_top_vec_);
    accum_loss += this->blob_top_loss_->cpu_data()[0];
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(4 * full_loss, accum_loss, 1e-3);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestForwardPixelWiseLossWeightZero) {
  // fill weights all with zero and check if loss is zero
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  for (int i = 0; i < this->blob_bottom_pixel_loss_weight_->count(); ++i) {
    this->blob_bottom_pixel_loss_weight_->mutable_cpu_data()[i] = 0;
  }
  layer.SetUp(this->blob_bottom_vec_3_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_3_, this->blob_top_vec_);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], 0, 1e-10);
}


}  // namespace caffe

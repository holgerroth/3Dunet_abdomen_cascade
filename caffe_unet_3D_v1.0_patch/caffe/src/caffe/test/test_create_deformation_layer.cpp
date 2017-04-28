#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/create_deformation_layer.hpp"
#include "caffe/util/vector_helper.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class CreateDeformationLayerTest : public ::testing::Test {
 protected:
  CreateDeformationLayerTest()
      : blob_top_a_(new Blob<Dtype>()),
        blob_top_b_(new Blob<Dtype>()),
        blob_top_c_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_top_vec_.push_back(blob_top_a_);
    blob_top_vec_.push_back(blob_top_b_);
    blob_top_vec_.push_back(blob_top_c_);
  }

  virtual ~CreateDeformationLayerTest() {
    delete blob_top_a_;
    delete blob_top_b_;
    delete blob_top_c_;
  }

  Blob<Dtype>* const blob_top_a_;
  Blob<Dtype>* const blob_top_b_;
  Blob<Dtype>* const blob_top_c_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CreateDeformationLayerTest, TestDtypes);

TYPED_TEST(CreateDeformationLayerTest, Test3DIdentityTransform) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  CreateDeformationParameter* create_deformation_param = param.mutable_create_deformation_param();
  create_deformation_param->set_batch_size(1);
  create_deformation_param->set_nz(3);
  create_deformation_param->set_ny(4);
  create_deformation_param->set_nx(5);
  create_deformation_param->set_ncomponents(3);
  this->blob_top_vec_.resize(1);
  CreateDeformationLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_a_->shape(0), 1);
  EXPECT_EQ(this->blob_top_a_->shape(1), 3);
  EXPECT_EQ(this->blob_top_a_->shape(2), 4);
  EXPECT_EQ(this->blob_top_a_->shape(3), 5);
  EXPECT_EQ(this->blob_top_a_->shape(4), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(
      "0,0,0,0,0,1,0,0,2,0,0,3,0,0,4,\n"
      "0,1,0,0,1,1,0,1,2,0,1,3,0,1,4,\n"
      "0,2,0,0,2,1,0,2,2,0,2,3,0,2,4,\n"
      "0,3,0,0,3,1,0,3,2,0,3,3,0,3,4,\n",
      Array2DtoString( this->blob_top_vec_[0]->cpu_data(), 4,5*3));
  EXPECT_EQ(
      "1,0,0,1,0,1,1,0,2,1,0,3,1,0,4,\n"
      "1,1,0,1,1,1,1,1,2,1,1,3,1,1,4,\n"
      "1,2,0,1,2,1,1,2,2,1,2,3,1,2,4,\n"
      "1,3,0,1,3,1,1,3,2,1,3,3,1,3,4,\n",
      Array2DtoString( this->blob_top_vec_[0]->cpu_data() + 1*4*5*3, 4,5*3));
  EXPECT_EQ(
      "2,0,0,2,0,1,2,0,2,2,0,3,2,0,4,\n"
      "2,1,0,2,1,1,2,1,2,2,1,3,2,1,4,\n"
      "2,2,0,2,2,1,2,2,2,2,2,3,2,2,4,\n"
      "2,3,0,2,3,1,2,3,2,2,3,3,2,3,4,\n",
      Array2DtoString( this->blob_top_vec_[0]->cpu_data() + 2*4*5*3, 4,5*3));

//  for (int z = 0; z < 10; ++z) {
//    for (int y = 0; y < 15; ++y) {
//      for (int x = 0; x < 20; ++x) {
//        EXPECT_EQ( z, this->blob_top_vec_[0]->data_at( make_vec(0,z,y,x,0)));
//        EXPECT_EQ( y, this->blob_top_vec_[0]->data_at( make_vec(0,z,y,x,1)));
//        EXPECT_EQ( x, this->blob_top_vec_[0]->data_at( make_vec(0,z,y,x,2)));
//      }
//    }
//  }
}

TYPED_TEST(CreateDeformationLayerTest, Test2DElasticTransform) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  CreateDeformationParameter* create_deformation_param = param.mutable_create_deformation_param();
  //  X - - - X - - - X - - - X - - - X - - - X - - - X
  //  - - - - - - - - - - - - - - - - - - - - - - - - -
  //  - - - - - - - - - - - - - - - - - - - - - - - - -
  //  - - - - - - - - - - - - - - - - - - - - - - - - -
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  - - - - # # # # # # # # # # # # # # # - - - - - -
  //  - - - - # # # # # # # # # # # # # # # - - - - - -
  //  - - - - # # # # # # # # # # # # # # # - - - - - -
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  - - - - # # # # # # # # # # # # # # # - - - - - -
  //  - - - - # # # # # # # # # # # # # # # - - - - - -
  //  - - - - # # # # # # # # # # # # # # # - - - - - -
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  - - - - # # # # # # # # # # # # # # # - - - - - -
  //  - - - - - - - - - - - - - - - - - - - - - - - - -
  //  - - - - - - - - - - - - - - - - - - - - - - - - -
  //  X - - - X - - - X - - - X - - - X - - - X - - - X
  //  - - - - - - - - - - - - - - - - - - - - - - - - -
  //  - - - - - - - - - - - - - - - - - - - - - - - - -
  //  - - - - - - - - - - - - - - - - - - - - - - - - -
  //  X - - - X - - - X - - - X - - - X - - - X - - - X
  //
  //
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  X - - - X # # # X # # # X # # # X # # - X - - - X
  //  X - - - X # # # X # # # X # # # X # # - X - - - X


  create_deformation_param->set_batch_size(3);
  create_deformation_param->set_ny(10);
  create_deformation_param->set_nx(15);
  create_deformation_param->set_ncomponents(2);
  create_deformation_param->mutable_random_elastic_grid_spacing()->add_v(4);
  create_deformation_param->mutable_random_elastic_grid_spacing()->add_v(4);
  create_deformation_param->mutable_random_elastic_deform_magnitude()->add_v(10);
  create_deformation_param->mutable_random_elastic_deform_magnitude()->add_v(10);
  this->blob_top_vec_.resize(1);
  CreateDeformationLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // fill blob with 42
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    this->blob_top_vec_[0]->mutable_cpu_data()[i] = 42;
  }

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // every should have been altered
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    EXPECT_NE( 42, this->blob_top_vec_[0]->cpu_data()[i])
        << "at index i=" << i;
  }
}

TYPED_TEST(CreateDeformationLayerTest, Test3DElasticTransform) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  CreateDeformationParameter* create_deformation_param = param.mutable_create_deformation_param();

  create_deformation_param->set_batch_size(3);
  create_deformation_param->set_nz(5);  // 5 control points
  create_deformation_param->set_ny(10); // 6 control points
  create_deformation_param->set_nx(15); // 7 control points
  create_deformation_param->set_ncomponents(3);
  create_deformation_param->mutable_random_elastic_grid_spacing()->add_v(3);
  create_deformation_param->mutable_random_elastic_grid_spacing()->add_v(4);
  create_deformation_param->mutable_random_elastic_grid_spacing()->add_v(4);
  create_deformation_param->mutable_random_elastic_deform_magnitude()->add_v(5);  create_deformation_param->mutable_random_elastic_deform_magnitude()->add_v(10);
  create_deformation_param->mutable_random_elastic_deform_magnitude()->add_v(10);
  this->blob_top_vec_.resize(1);
  CreateDeformationLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // fill blob with 42
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    this->blob_top_vec_[0]->mutable_cpu_data()[i] = 42;
  }

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // every should have been altered
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    EXPECT_NE( 42, this->blob_top_vec_[0]->cpu_data()[i])
        << "at index i=" << i;
  }
}

TYPED_TEST(CreateDeformationLayerTest, TestUnitTransform) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  CreateDeformationParameter* create_deformation_param = param.mutable_create_deformation_param();

  create_deformation_param->set_batch_size(1);
  create_deformation_param->set_nz(3);
  create_deformation_param->set_ny(4);
  create_deformation_param->set_nx(5);
  create_deformation_param->set_ncomponents(3);
  create_deformation_param->mutable_random_elastic_grid_spacing()->add_v(1);
  create_deformation_param->mutable_random_elastic_grid_spacing()->add_v(1);
  create_deformation_param->mutable_random_elastic_grid_spacing()->add_v(1);
  create_deformation_param->mutable_random_elastic_deform_magnitude()->add_v(1e-30);
  create_deformation_param->mutable_random_elastic_deform_magnitude()->add_v(1e-30);
  create_deformation_param->mutable_random_elastic_deform_magnitude()->add_v(1e-30);
  this->blob_top_vec_.resize(1);
  CreateDeformationLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // fill blob with 42
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    this->blob_top_vec_[0]->mutable_cpu_data()[i] = 42;
  }

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // check for unity
  EXPECT_EQ(
      "0,0,0,0,0,1,0,0,2,0,0,3,0,0,4,\n"
      "0,1,0,0,1,1,0,1,2,0,1,3,0,1,4,\n"
      "0,2,0,0,2,1,0,2,2,0,2,3,0,2,4,\n"
      "0,3,0,0,3,1,0,3,2,0,3,3,0,3,4,\n",
      Array2DtoString( this->blob_top_vec_[0]->cpu_data(), 4, 5*3));
  EXPECT_EQ(
      "1,0,0,1,0,1,1,0,2,1,0,3,1,0,4,\n"
      "1,1,0,1,1,1,1,1,2,1,1,3,1,1,4,\n"
      "1,2,0,1,2,1,1,2,2,1,2,3,1,2,4,\n"
      "1,3,0,1,3,1,1,3,2,1,3,3,1,3,4,\n",
      Array2DtoString( this->blob_top_vec_[0]->cpu_data() + 1*4*5*3, 4,5*3));
  EXPECT_EQ(
      "2,0,0,2,0,1,2,0,2,2,0,3,2,0,4,\n"
      "2,1,0,2,1,1,2,1,2,2,1,3,2,1,4,\n"
      "2,2,0,2,2,1,2,2,2,2,2,3,2,2,4,\n"
      "2,3,0,2,3,1,2,3,2,2,3,3,2,3,4,\n",
      Array2DtoString( this->blob_top_vec_[0]->cpu_data() + 2*4*5*3, 4,5*3));
}


TYPED_TEST(CreateDeformationLayerTest, TestAffineTransform) {
  vector<TypeParam> M = make_vec<TypeParam>( 3, 0, 0,
                                             2, 0, 0,
                                             1, 0, 0);
  vector<TypeParam> v = make_vec<TypeParam>( 1, 0, 0);
  vector<TypeParam> out(3);

  caffe_cpu_gemv<TypeParam>(CblasNoTrans, 3, 3, 1, M.data(), v.data(), 1, out.data());
  EXPECT_EQ( 3, out[0]);
  EXPECT_EQ( 2, out[1]);
  EXPECT_EQ( 1, out[2]);

  // TODO: continue this test

}

TYPED_TEST(CreateDeformationLayerTest, TestProbMapSampling) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  CreateDeformationParameter* create_deformation_param = param.mutable_create_deformation_param();

  create_deformation_param->set_nz(7);
  create_deformation_param->set_ny(8);
  create_deformation_param->set_nx(9);
  create_deformation_param->set_ncomponents(3);
  create_deformation_param->set_random_offset_range_from_pdf(true);

  Blob<TypeParam> probMap( make_vec( 3,1,7,8,9));
  caffe_set( probMap.count(), TypeParam(0), probMap.mutable_cpu_data());

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back( &probMap);

  probMap.mutable_cpu_data()[ probMap.offset( make_vec(0,0,3,2,6))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(1,0,0,0,0))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(2,0,6,7,8))] = 1;

  this->blob_top_vec_.resize(1);
  CreateDeformationLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // fill output blob with 42
  caffe_set(this->blob_top_vec_[0]->count(), TypeParam(42),
            this->blob_top_vec_[0]->mutable_cpu_data());

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const TypeParam* data = this->blob_top_vec_[0]->cpu_data();
  EXPECT_EQ( "(0,-1.5,2)", toString( data, 3));
  EXPECT_EQ( "(-3,-3.5,-4)", toString( data+7*8*9*3, 3));
  EXPECT_EQ( "(3,3.5,4)", toString( data+2*7*8*9*3, 3));

  // other positions
  caffe_set( probMap.count(), TypeParam(0), probMap.mutable_cpu_data());
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(0,0,2,5,3))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(1,0,3,2,1))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(2,0,0,0,0))] = 1;
  caffe_set(this->blob_top_vec_[0]->count(), TypeParam(42),
            this->blob_top_vec_[0]->mutable_cpu_data());
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  data = this->blob_top_vec_[0]->cpu_data();
  EXPECT_EQ( "(-1,1.5,-1)", toString( data, 3));
  EXPECT_EQ( "(0,-1.5,-3)", toString( data+7*8*9*3, 3));
  EXPECT_EQ( "(-3,-3.5,-4)", toString( data+2*7*8*9*3, 3));
}

TYPED_TEST(CreateDeformationLayerTest, TestProbMapSampling2) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  CreateDeformationParameter* create_deformation_param = param.mutable_create_deformation_param();

  // same as test above, but for a smaller output
  create_deformation_param->set_nz(1);
  create_deformation_param->set_ny(5);
  create_deformation_param->set_nx(3);
  create_deformation_param->set_ncomponents(3);
  create_deformation_param->set_random_offset_range_from_pdf(true);

  Blob<TypeParam> probMap( make_vec( 3,1,7,8,9));
  caffe_set( probMap.count(), TypeParam(0), probMap.mutable_cpu_data());

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back( &probMap);

  probMap.mutable_cpu_data()[ probMap.offset( make_vec(0,0,3,2,6))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(1,0,0,0,0))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(2,0,6,7,8))] = 1;

  this->blob_top_vec_.resize(1);
  CreateDeformationLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // fill output blob with 42
  caffe_set(this->blob_top_vec_[0]->count(), TypeParam(42),
            this->blob_top_vec_[0]->mutable_cpu_data());

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const TypeParam* data = this->blob_top_vec_[0]->cpu_data();
  EXPECT_EQ( "(3,0,5)", toString( data, 3));
  EXPECT_EQ( "(0,-2,-1)", toString( data+1*5*3*3, 3));
  EXPECT_EQ( "(6,5,7)", toString( data+2*1*5*3*3, 3));

  // other positions
  caffe_set( probMap.count(), TypeParam(0), probMap.mutable_cpu_data());
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(0,0,2,5,3))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(1,0,3,2,1))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(2,0,0,0,0))] = 1;
  caffe_set(this->blob_top_vec_[0]->count(), TypeParam(42),
            this->blob_top_vec_[0]->mutable_cpu_data());
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  data = this->blob_top_vec_[0]->cpu_data();
  EXPECT_EQ( "(2,3,2)", toString( data, 3));
  EXPECT_EQ( "(3,0,0)", toString( data+1*5*3*3, 3));
  EXPECT_EQ( "(0,-2,-1)", toString( data+2*1*5*3*3, 3));
}


TYPED_TEST(CreateDeformationLayerTest, TestProbMapSampling3) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  CreateDeformationParameter* create_deformation_param = param.mutable_create_deformation_param();

  // now for 2D data
  create_deformation_param->set_ny(5);
  create_deformation_param->set_nx(3);
  create_deformation_param->set_ncomponents(2);
  create_deformation_param->set_random_offset_range_from_pdf(true);

  Blob<TypeParam> probMap( make_vec( 3,1,8,9));
  caffe_set( probMap.count(), TypeParam(0), probMap.mutable_cpu_data());

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back( &probMap);

  probMap.mutable_cpu_data()[ probMap.offset( make_vec(0,0,2,6))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(1,0,0,0))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(2,0,7,8))] = 1;

  this->blob_top_vec_.resize(1);
  CreateDeformationLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // fill output blob with 42
  caffe_set(this->blob_top_vec_[0]->count(), TypeParam(42),
            this->blob_top_vec_[0]->mutable_cpu_data());

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const TypeParam* data = this->blob_top_vec_[0]->cpu_data();
  EXPECT_EQ( "(0,5)", toString( data, 2));
  EXPECT_EQ( "(-2,-1)", toString( data+1*5*3*2, 2));
  EXPECT_EQ( "(5,7)", toString( data+2*1*5*3*2, 2));

  // other positions
  caffe_set( probMap.count(), TypeParam(0), probMap.mutable_cpu_data());
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(0,0,5,3))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(1,0,2,1))] = 1;
  probMap.mutable_cpu_data()[ probMap.offset( make_vec(2,0,0,0))] = 1;
  caffe_set(this->blob_top_vec_[0]->count(), TypeParam(42),
            this->blob_top_vec_[0]->mutable_cpu_data());
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  data = this->blob_top_vec_[0]->cpu_data();
  EXPECT_EQ( "(3,2)", toString( data, 2));
  EXPECT_EQ( "(0,0)", toString( data+1*5*3*2, 2));
  EXPECT_EQ( "(-2,-1)", toString( data+2*1*5*3*2, 2));
}

TYPED_TEST(CreateDeformationLayerTest, TestIgnoreLabelSampling) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  CreateDeformationParameter* create_deformation_param = param.mutable_create_deformation_param();

  create_deformation_param->set_nz(1);
  create_deformation_param->set_ny(5);
  create_deformation_param->set_nx(3);
  create_deformation_param->set_ncomponents(3);
  create_deformation_param->set_random_offset_range_from_ignore_label(7);

  Blob<TypeParam> labelMap( make_vec( 3,1,7,8,9));
  caffe_set( labelMap.count(), TypeParam(7), labelMap.mutable_cpu_data());

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back( &labelMap);

  labelMap.mutable_cpu_data()[ labelMap.offset( make_vec(0,0,3,2,6))] = 2;
  labelMap.mutable_cpu_data()[ labelMap.offset( make_vec(1,0,0,0,0))] = 3;
  labelMap.mutable_cpu_data()[ labelMap.offset( make_vec(2,0,6,7,8))] = 4;

  this->blob_top_vec_.resize(1);
  CreateDeformationLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // fill output blob with 42
  caffe_set(this->blob_top_vec_[0]->count(), TypeParam(42),
            this->blob_top_vec_[0]->mutable_cpu_data());

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const TypeParam* data = this->blob_top_vec_[0]->cpu_data();
  EXPECT_EQ( "(3,0,5)", toString( data, 3));
  EXPECT_EQ( "(0,-2,-1)", toString( data+1*5*3*3, 3));
  EXPECT_EQ( "(6,5,7)", toString( data+2*1*5*3*3, 3));

  // other positions
  caffe_set( labelMap.count(), TypeParam(7), labelMap.mutable_cpu_data());
  labelMap.mutable_cpu_data()[ labelMap.offset( make_vec(0,0,2,5,3))] = 1;
  labelMap.mutable_cpu_data()[ labelMap.offset( make_vec(1,0,3,2,1))] = 6;
  labelMap.mutable_cpu_data()[ labelMap.offset( make_vec(2,0,0,0,0))] = 5;
  caffe_set(this->blob_top_vec_[0]->count(), TypeParam(42),
            this->blob_top_vec_[0]->mutable_cpu_data());
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  data = this->blob_top_vec_[0]->cpu_data();
  EXPECT_EQ( "(2,3,2)", toString( data, 3));
  EXPECT_EQ( "(3,0,0)", toString( data+1*5*3*3, 3));
  EXPECT_EQ( "(0,-2,-1)", toString( data+2*1*5*3*3, 3));
}

TYPED_TEST(CreateDeformationLayerTest, TestSamplingInBlobShape) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  CreateDeformationParameter* create_deformation_param = param.mutable_create_deformation_param();

  int out_nz = 5;
  int out_ny = 7;
  int out_nx = 3;
  int out_ncomp = 3;

  create_deformation_param->set_nz(out_nz);
  create_deformation_param->set_ny(out_ny);
  create_deformation_param->set_nx(out_nx);
  create_deformation_param->set_ncomponents(out_ncomp);
  create_deformation_param->set_random_offset_range_from_in_blob_shape(true);

  int in_nz = 7;
  int in_ny = 8;
  int in_nx = 9;
  Blob<TypeParam> inBlob( make_vec( 3, 1, in_nz, in_ny, in_nx));
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back( &inBlob);

  this->blob_top_vec_.resize(1);
  CreateDeformationLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // fill output blob with 42
  caffe_set(this->blob_top_vec_[0]->count(), TypeParam(42),
            this->blob_top_vec_[0]->mutable_cpu_data());

  // check for several iterations, if the central pixel of the output
  // blob is always within the input blob
  const TypeParam* centralOutPixel = this->blob_top_vec_[0]->cpu_data()
      + ( ( ( (out_nz-1)/2 * out_ny)
            + (out_ny-1)/2) * out_nx
          + (out_nx-1)/2) * out_ncomp;
  for(int iter = 0; iter < 1000; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    //   std::cout << "pos: " << toString( centralOutPixel, 3) << std::endl;
    EXPECT_GE( centralOutPixel[0], 0);
    EXPECT_LT( centralOutPixel[0], in_nz);
    EXPECT_GE( centralOutPixel[1], 0);
    EXPECT_LT( centralOutPixel[1], in_ny);
    EXPECT_GE( centralOutPixel[2], 0);
    EXPECT_LT( centralOutPixel[2], in_nx);
  }
}




}  // namespace caffe

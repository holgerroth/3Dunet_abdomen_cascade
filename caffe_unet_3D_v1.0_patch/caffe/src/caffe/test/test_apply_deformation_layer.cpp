#include <string>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"
#include "caffe/layers/apply_deformation_layer.hpp"
#include "caffe/util/vector_helper.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ApplyDeformationLayerTest : public ::testing::Test {
 protected:
  ApplyDeformationLayerTest() {
  }

  virtual void SetUp() {
  }

  virtual ~ApplyDeformationLayerTest() {
  }

};

TYPED_TEST_CASE(ApplyDeformationLayerTest, TestDtypes);

TYPED_TEST(ApplyDeformationLayerTest, TestOutShape) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  ApplyDeformationParameter* apply_deformation_param = param.mutable_apply_deformation_param();
  apply_deformation_param->set_interpolation("linear");
  apply_deformation_param->set_extrapolation("mirror");

  // Test reshaping for 2 input blobs
  // 2D, 2 components
  ApplyDeformationLayer<TypeParam> layer(param);
  Blob<TypeParam> data;
  Blob<TypeParam> deform;
  data.Reshape(2, 4, 5, 7);  // num, c, y, x
  deform.Reshape(2, 5, 7, 2); // num, y, x, ncomp
  vector<Blob<TypeParam>*> blob_bottom_vec;
  blob_bottom_vec.push_back( &data);
  blob_bottom_vec.push_back( &deform);
  Blob<TypeParam> top;
  vector<Blob<TypeParam>*> blob_top_vec;
  blob_top_vec.push_back( &top);
  layer.SetUp(blob_bottom_vec, blob_top_vec);

  EXPECT_EQ("2 4 5 7 (280)", top.shape_string());

  // 3D, 2 components
  data.Reshape(make_vec<int>(2, 4, 5, 7, 9));  // num, c, z, y, x
  deform.Reshape(make_vec<int>(2, 5, 7, 9, 2)); // num, z, y, x, ncomp
  layer.SetUp(blob_bottom_vec, blob_top_vec);
  EXPECT_EQ("2 4 5 7 9 (2520)", top.shape_string());

  // 3D, 3 components
  data.Reshape(make_vec<int>(2, 4, 5, 7, 9));  // num, c, z, y, x
  deform.Reshape(make_vec<int>(2, 5, 7, 9, 3)); // num, z, y, x, ncomp
  layer.SetUp(blob_bottom_vec, blob_top_vec);
  EXPECT_EQ("2 4 5 7 9 (2520)", top.shape_string());

#if 0
  // 3D, 2 components
  data.Reshape(make_vec<int>(2, 4, 5, 7, 9));  // num, c, z, y, x
  deform.Reshape(make_vec<int>(2, 5, 7, 9, 2)); // num, z, y, x, ncomp
  ref.Reshape(make_vec<int>(2, 1, 3, 4, 6)); // num, c, z, y, x  (only shape of z, y and x is used)
  layer.SetUp(blob_bottom_vec, blob_top_vec);
  EXPECT_EQ("2 4 3 4 6 (576)", top.shape_string());

  // 3D, 3 components
  deform.Reshape(make_vec<int>(2, 5, 7, 9, 3)); // num, z, y, x, ncomp
  layer.SetUp(blob_bottom_vec, blob_top_vec);
  EXPECT_EQ("2 4 3 4 6 (576)", top.shape_string());
#endif
}

TYPED_TEST(ApplyDeformationLayerTest, TestOutShapeFromRef_2D_2comp) {
  // Test reshaping to shape of a reference blob
  // 2D, 2 components
  string proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'DummyData' "
      "  dummy_data_param { "
      "    shape { dim: 2 dim: 4 dim: 5 dim: 7 } " // num, c, y, x
      "    shape { dim: 2 dim: 5 dim: 7 dim: 2 } " // num, y, x, ncomp
      "    shape { dim: 2 dim: 1 dim: 3 dim: 4 } " // num, c, y, x (only
      "  } "                                       //  shape of y and x is used)
      "  top: 'data' "
      "  top: 'deform' "
      "  top: 'ref' "
      "} "
      "layer { "
      "  name: 'apply_deformation' "
      "  type: 'ApplyDeformation' "
      "  bottom: 'data' "
      "  bottom: 'deform' "
      "  top: 'deformed_data' "
      "  apply_deformation_param { "
      "    interpolation: 'nearest' "
      "    extrapolation: 'mirror' "
      "    output_shape_from: 'ref' "
      "  } "
      "}";
  NetParameter param;
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
  Net<TypeParam> net( param);
  EXPECT_EQ("2 4 3 4 (96)", net.blob_by_name("deformed_data")->shape_string());
}

TYPED_TEST(ApplyDeformationLayerTest, TestOutShapeFromRef_3D_2comp) {
  // Test reshaping to shape of a reference blob
  // 2D, 2 components
  string proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'DummyData' "
      "  dummy_data_param { "
      "    shape { dim: 2 dim: 4 dim: 5 dim: 7 dim: 9} " // num, c, z, y, x
      "    shape { dim: 2 dim: 5 dim: 7 dim: 9 dim: 2 } " // num, z, y, x, ncomp
      "    shape { dim: 2 dim: 1 dim: 3 dim: 4 dim: 6 } " // num, c, z, y, x (only shape of z, y and x is used)
      "  } "
      "  top: 'data' "
      "  top: 'deform' "
      "  top: 'ref' "
      "} "
      "layer { "
      "  name: 'apply_deformation' "
      "  type: 'ApplyDeformation' "
      "  bottom: 'data' "
      "  bottom: 'deform' "
      "  top: 'deformed_data' "
      "  apply_deformation_param { "
      "    interpolation: 'nearest' "
      "    extrapolation: 'mirror' "
      "    output_shape_from: 'ref' "
      "  } "
      "}";
  NetParameter param;
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
  Net<TypeParam> net( param);
  EXPECT_EQ("2 4 3 4 6 (576)", net.blob_by_name("deformed_data")->shape_string());
}

TYPED_TEST(ApplyDeformationLayerTest, TestOutShapeFromRef_3D_3comp) {
  // Test reshaping to shape of a reference blob
  // 2D, 2 components
  string proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'DummyData' "
      "  dummy_data_param { "
      "    shape { dim: 2 dim: 4 dim: 5 dim: 7 dim: 9} " // num, c, z, y, x
      "    shape { dim: 2 dim: 5 dim: 7 dim: 9 dim: 3 } " // num, z, y, x, ncomp
      "    shape { dim: 2 dim: 1 dim: 3 dim: 4 dim: 6 } " // num, c, z, y, x (only shape of z, y and x is used)
      "  } "
      "  top: 'data' "
      "  top: 'deform' "
      "  top: 'ref' "
      "} "
      "layer { "
      "  name: 'apply_deformation' "
      "  type: 'ApplyDeformation' "
      "  bottom: 'data' "
      "  bottom: 'deform' "
      "  top: 'deformed_data' "
      "  apply_deformation_param { "
      "    interpolation: 'nearest' "
      "    extrapolation: 'mirror' "
      "    output_shape_from: 'ref' "
      "  } "
      "}";
  NetParameter param;
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
  Net<TypeParam> net( param);
  EXPECT_EQ("2 4 3 4 6 (576)", net.blob_by_name("deformed_data")->shape_string());
}

TYPED_TEST(ApplyDeformationLayerTest, TestExtrapolationMirror) {
  Caffe::set_mode(Caffe::CPU);
  TypeParam data[] = {0,1,2,3,4};
  TypeParam def[] = {0,-10,
                     0,-9,
                     0,-8,
                     0,-7,
                     0,-6,
                     0,-5,
                     0,-4,
                     0,-3,
                     0,-2,
                     0,-1,
                     0,0,
                     0,1,
                     0,2,
                     0,3,
                     0,4,
                     0,5,
                     0,6,
                     0,7,
                     0,8,
                     0,9,
                     0,10};
  std::vector<TypeParam> out(21);
  transform2D( data, 1, 1, 1, 1,  5,
               def,  1,    1, 1, 21,
               out.data(),  1, 1, 1, 1, 21, linear2D_mirror<TypeParam>);
  EXPECT_EQ( "(2,1,0,1,2,3,4,3,2,1,0,1,2,3,4,3,2,1,0,1,2)", toString(out));
}

TYPED_TEST(ApplyDeformationLayerTest, TestExtrapolationZero) {
  Caffe::set_mode(Caffe::CPU);
  TypeParam data[] = {10,11,12,13,14};
  TypeParam def[] = {0,-10,
                     0,-9,
                     0,-8,
                     0,-7,
                     0,-6,
                     0,-5,
                     0,-4,
                     0,-3,
                     0,-2,
                     0,-1,
                     0,0,
                     0,1,
                     0,2,
                     0,3,
                     0,4,
                     0,5,
                     0,6,
                     0,7,
                     0,8,
                     0,9,
                     0,10};
  std::vector<TypeParam> out(21);
  transform2D( data, 1, 1, 1, 1,  5,
               def,  1,    1, 1, 21,
               out.data(),  1, 1, 1, 1, 21, linear2D_zeropad<TypeParam>);
  EXPECT_EQ( "(0,0,0,0,0,0,0,0,0,0,10,11,12,13,14,0,0,0,0,0,0)", toString(out));
}

TYPED_TEST(ApplyDeformationLayerTest, TestLinearInterpolation2D) {
  Caffe::set_mode(Caffe::CPU);
  TypeParam data[] = {0, 1,
                      0, 2};
  TypeParam def[] = {0,  0.5,
                     0,  1,
                     0.5, 0.5,
                     0.5, 1};
  std::vector<TypeParam> out(4);
  transform2D( data, 1, 1, 1, 2, 2,
               def,  1,    1, 2, 2,
               out.data(),  1, 1, 1, 2, 2, linear2D_zeropad<TypeParam>);
  EXPECT_EQ( "(0.5,1,0.75,1.5)", toString(out));
  transform2D( data, 1, 1, 1, 2, 2,
               def,  1,    1, 2, 2,
               out.data(),  1, 1, 1, 2, 2, linear2D_mirror<TypeParam>);
  EXPECT_EQ( "(0.5,1,0.75,1.5)", toString(out));
}

TYPED_TEST(ApplyDeformationLayerTest, TestLinearInterpolation3D) {
  Caffe::set_mode(Caffe::CPU);
  TypeParam data[] = {0, 1,
                      0, 2,
                      0, 0,
                      0, 3};
  TypeParam def[] = {0,   0, 0.5,
                     0,   1,   1,
                     0.5, 0.5, 0.5,
                     1,   1,   0.25};
  std::vector<TypeParam> out(4);
  transform3D( data, 1, 1, 2, 2, 2,
               def,  1,    1, 2, 2,
               out.data(),  1, 1, 1, 2, 2, linear3D_zeropad<TypeParam>);
  EXPECT_EQ( "(0.5,2,0.75,0.75)", toString(out));
  transform3D( data, 1, 1, 2, 2, 2,
               def,  1,    1, 2, 2,
               out.data(),  1, 1, 1, 2, 2, linear3D_mirror<TypeParam>);
  EXPECT_EQ( "(0.5,2,0.75,0.75)", toString(out));
}

TYPED_TEST(ApplyDeformationLayerTest, TestNearestInterpolation2D) {
  Caffe::set_mode(Caffe::CPU);
  TypeParam data[] = {0, 1,
                      0, 2};
  TypeParam def[] = {0,  0.5,
                     0,  1,
                     0.5, 0.5,
                     0.5, 1};
  std::vector<TypeParam> out(4);
  transform2D( data, 1, 1, 1, 2, 2,
               def,  1,    1, 2, 2,
               out.data(),  1, 1, 1, 2, 2, nearest2D_zeropad<TypeParam>);
  EXPECT_EQ( "(1,1,2,2)", toString(out));
  transform2D( data, 1, 1, 1, 2, 2,
               def,  1,    1, 2, 2,
               out.data(),  1, 1, 1, 2, 2, nearest2D_mirror<TypeParam>);
  EXPECT_EQ( "(1,1,2,2)", toString(out));
}

TYPED_TEST(ApplyDeformationLayerTest, TestnearestInterpolation3D) {
  Caffe::set_mode(Caffe::CPU);
  TypeParam data[] = {0, 1,
                      0, 2,
                      0, 0,
                      0, 3};
  TypeParam def[] = {0,   0, 0.5,
                     0,   1,   1,
                     0.5, 0.5, 0.5,
                     1,   1,   0.25};
  std::vector<TypeParam> out(4);
  transform3D( data, 1, 1, 2, 2, 2,
               def,  1,    1, 2, 2,
               out.data(),  1, 1, 1, 2, 2, nearest3D_zeropad<TypeParam>);
  EXPECT_EQ( "(1,2,3,0)", toString(out));
  transform3D( data, 1, 1, 2, 2, 2,
               def,  1,    1, 2, 2,
               out.data(),  1, 1, 1, 2, 2, nearest3D_mirror<TypeParam>);
  EXPECT_EQ( "(1,2,3,0)", toString(out));
}

TYPED_TEST(ApplyDeformationLayerTest, TestRotate90) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  ApplyDeformationParameter* apply_deformation_param = param.mutable_apply_deformation_param();
  apply_deformation_param->set_interpolation("linear");
  apply_deformation_param->set_extrapolation("zero");

  // Test 90 degree rotation
  // 2D, 2 components
  ApplyDeformationLayer<TypeParam> layer(param);
  Blob<TypeParam> data;
  Blob<TypeParam> deform;
  data.Reshape(2, 4, 5, 9);  // num, c, y, x
  deform.Reshape(2, 5, 9, 2); // num, y, x, ncomp
  vector<Blob<TypeParam>*> blob_bottom_vec;
  blob_bottom_vec.push_back( &data);
  blob_bottom_vec.push_back( &deform);
  Blob<TypeParam> top;
  vector<Blob<TypeParam>*> blob_top_vec;
  blob_top_vec.push_back( &top);
  layer.SetUp(blob_bottom_vec, blob_top_vec);

  // fill data blob with unique values
  for( int i = 0; i < data.count(); ++i) {
    data.mutable_cpu_data()[i] = 10 + i;
  }
//  std::cout << "input field: \n"
//            << Array2DtoString( data.cpu_data(), data.shape(-2), data.shape(-1))
//            << std::endl;

  // setup 90 degree rotation
  for( int n = 0; n < deform.shape(0); ++n) {
    for( int y = 0; y < deform.shape(1); ++y) {
      for( int x = 0; x < deform.shape(2); ++x) {
        int offs = deform.offset(make_vec<int>(n, y, x, 0));
        deform.mutable_cpu_data()[offs] = x - 2;
        deform.mutable_cpu_data()[offs + 1] = 6 - y;
      }
    }
  }
//  std::cout << "Deformation field: \n"
//            << Array2DtoString( deform.cpu_data(), deform.shape(1), deform.shape(2)*2)
//            << std::endl;
  // apply layer
  layer.Forward(blob_bottom_vec, blob_top_vec);

  // check result
//  std::cout << "output field: \n"
//            << Array2DtoString( top.cpu_data(), top.shape(-2), top.shape(-1))
//            << std::endl;

  EXPECT_EQ(
      "0,0,16,25,34,43,52,0,0,\n"
      "0,0,15,24,33,42,51,0,0,\n"
      "0,0,14,23,32,41,50,0,0,\n"
      "0,0,13,22,31,40,49,0,0,\n"
      "0,0,12,21,30,39,48,0,0,\n", Array2DtoString( top.cpu_data(), top.shape(-2), top.shape(-1)));

  EXPECT_EQ(
      "0,0,331,340,349,358,367,0,0,\n"
      "0,0,330,339,348,357,366,0,0,\n"
      "0,0,329,338,347,356,365,0,0,\n"
      "0,0,328,337,346,355,364,0,0,\n"
      "0,0,327,336,345,354,363,0,0,\n", Array2DtoString( top.cpu_data() + top.offset(1,3,0,0), top.shape(-2), top.shape(-1)));

  // test mirroring
  //
  apply_deformation_param->set_extrapolation("mirror");
  ApplyDeformationLayer<TypeParam> layer2(param);
  layer2.SetUp(blob_bottom_vec, blob_top_vec);
  layer2.Forward(blob_bottom_vec, blob_top_vec);
  EXPECT_EQ(
      "34,25,16,25,34,43,52,43,34,\n"
      "33,24,15,24,33,42,51,42,33,\n"
      "32,23,14,23,32,41,50,41,32,\n"
      "31,22,13,22,31,40,49,40,31,\n"
      "30,21,12,21,30,39,48,39,30,\n", Array2DtoString( top.cpu_data(), top.shape(-2), top.shape(-1)));

}

}  // namespace caffe

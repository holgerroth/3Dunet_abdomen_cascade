#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/util/vector_helper.hpp"
#include "caffe/test/test_caffe_main.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template<typename Dtype>
class VectorHelperTest : public ::testing::Test {
 protected:
  VectorHelperTest(){
  }

  virtual void SetUp() {
  }

  virtual ~VectorHelperTest() {
  }

};

TYPED_TEST_CASE(VectorHelperTest, TestDtypes);

TYPED_TEST(VectorHelperTest, TestBasics) {
  vector<TypeParam> A = make_vec<TypeParam>(5,4,3,2,1);
  EXPECT_EQ( "(5,4,3,2,1)", toString(A));
}

TYPED_TEST(VectorHelperTest, TestShift3D) {
  vector<TypeParam> A = make_vec<TypeParam>(
      3,3,2,3,
      1,2,5,4,
      2,1,1,2,
      0,0,0,1);
  vector<TypeParam> B = m_shift3D<TypeParam>(7, 4, 12, A);

  vector<TypeParam> R = make_vec<TypeParam>(
      3,3,2,10,
      1,2,5,8,
      2,1,1,14,
      0,0,0,1);
  EXPECT_EQ( toString(R), toString(B));
}

TYPED_TEST(VectorHelperTest, TestScale3D) {
  // test scaling of unit matrix
  vector<TypeParam> A = make_vec<TypeParam>(
      1,0,0,0,
      0,1,0,0,
      0,0,1,0,
      0,0,0,1);
  vector<TypeParam> B = m_scale3D<TypeParam>(4, 3, 2, A);
  EXPECT_EQ( "(4,0,0,0,"
             "0,3,0,0,"
             "0,0,2,0,"
             "0,0,0,1)", toString(B));

  // test scaling forward and inverse
   vector<TypeParam> C = make_vec<TypeParam>(
      3,3,2,3,
      1,2,5,4,
      2,1,1,2,
      0,0,0,1);

   vector<TypeParam> D = m_scale3D<TypeParam>(4, 3, 2, C);
   vector<TypeParam> E = m_scale3D<TypeParam>(1.0/4, 1.0/3, 1.0/2, D);

   EXPECT_EQ( toString(E), toString(C));
}

TYPED_TEST(VectorHelperTest, TestRotate3D) {
  vector<TypeParam> A = make_vec<TypeParam>(
      4,0,0,0,
      0,3,0,0,
      0,0,2,0,
      0,0,0,1);

  // rotation around first axis
  vector<TypeParam> B = m_rotate3D<TypeParam>(90, 0, 0, A);
  for (int i = 0; i < 16; ++i) {
    B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
  }

  EXPECT_EQ( "(4,0,0,0,"
             "0,0,-2,0,"
             "0,3,0,0,"
             "0,0,0,1)", toString(B));

  // rotation around second axis
  B = m_rotate3D<TypeParam>(0, 90, 0, A);
  for (int i = 0; i < 16; ++i) {
    B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
  }

  EXPECT_EQ( "(0,0,-2,0,"
             "0,3,0,0,"
             "4,0,0,0,"
             "0,0,0,1)", toString(B));

  // rotation around third axis
  B = m_rotate3D<TypeParam>(0, 0, 90, A);
  for (int i = 0; i < 16; ++i) {
    B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
  }

  EXPECT_EQ( "(0,-3,0,0,"
             "4,0,0,0,"
             "0,0,2,0,"
             "0,0,0,1)", toString(B));

  // rotate forward backward
  vector<TypeParam> C = make_vec<TypeParam>(
      3,3,2,3,
      1,2,5,4,
      2,1,1,2,
      0,0,0,1);
  B = m_rotate3D<TypeParam>(27.5, 0, 0, C);
  B = m_rotate3D<TypeParam>(-27.5, 0, 0, B);

  for (int i = 0; i < 16; ++i) {
    B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
  }

  EXPECT_EQ( toString(B), toString(C));

  B = m_rotate3D<TypeParam>(0, 13.9, 0, C);
  B = m_rotate3D<TypeParam>(0, -13.9, 0, B);

  for (int i = 0; i < 16; ++i) {
    B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
  }

  EXPECT_EQ( toString(B), toString(C));

  B = m_rotate3D<TypeParam>(0, 0, 42.14, C);
  B = m_rotate3D<TypeParam>(0, 0, -42.14, B);

  for (int i = 0; i < 16; ++i) {
    B[i] = std::floor( B[i] * 100000 + 0.5) / 100000;
  }

  EXPECT_EQ( toString(B), toString(C));

}

TYPED_TEST(VectorHelperTest, TestFlip2D) {
  vector<TypeParam> A = make_vec<TypeParam>(
      3,3,2,3,
      1,2,5,4,
      2,7,1,2);
  vector<TypeParam> B = make_vec<TypeParam>(
	  3,2,3,3,
	  4,5,2,1,
	  2,1,7,2);
  vector<TypeParam> C = make_vec<TypeParam>(
	  2,7,1,2,
	  1,2,5,4,
	  3,3,2,3);
  vector<TypeParam> D = make_vec<TypeParam>(
	  2,1,7,2,
	  4,5,2,1,
	  3,2,3,3);
  vector<TypeParam> B2(A.size(), 0);
  vector<TypeParam> C2(A.size(), 0);
  vector<TypeParam> D2(A.size(), 0);
  vector<TypeParam> A2(A.size(), 0);

  const TypeParam* in = A.data();
  const int inNb = 1;
  const int inNc = 1;
  const int inNz = 1;
  const int inNy = 3;
  const int inNx = 4;

  TypeParam* out = B2.data();
  const int outNb = inNb;
  const int outNc = inNc;
  const int outNz = inNz;
  const int outNy = inNy;
  const int outNx = inNx;
  flip2D_x( in, inNb, inNc, inNz, inNy, inNx,
			out,outNb, outNc, outNz, outNy, outNx);

  EXPECT_EQ( toString(B), toString(B2));

  out = C2.data();
  flip2D_y( in, inNb, inNc, inNz, inNy, inNx,
			out,outNb, outNc, outNz, outNy, outNx);

  EXPECT_EQ( toString(C), toString(C2));

  out = D2.data();
  flip2D_xy( in, inNb, inNc, inNz, inNy, inNx,
			 out,outNb, outNc, outNz, outNy, outNx);

  EXPECT_EQ( toString(D), toString(D2));

  const TypeParam* in2 = B2.data();
  out = A2.data();
  flip2D_x( in2, inNb, inNc, inNz, inNy, inNx,
			out,outNb, outNc, outNz, outNy, outNx);

  EXPECT_EQ( toString(A), toString(A2));

}

TYPED_TEST(VectorHelperTest, TestCropAndFlip) {

  // 1D tests
  {
    // Crop within data range without flipping 1D
    int inSize = 14, outSize = 7, offset = 3;
    vector<TypeParam> in(inSize);
    for (int i = 0; i < in.size(); ++i) in[i] = i;
    vector<int> inShape(1, in.size());
    vector<int> offs(1, offset);
    vector<bool> flip(1, false);

    vector<TypeParam> expectedOut(outSize);
    for (int i = 0; i < expectedOut.size(); ++i)
        expectedOut[i] = in[i + offset];

    vector<TypeParam> out(expectedOut.size(), 0);
    vector<int> outShape(1, out.size());
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop within data range with flipping 1D
    for (int i = 0; i < expectedOut.size(); ++i)
        expectedOut[i] = in[outShape[0] - i - 1 + offset];
    flip[0] = true;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop partially within data range (left) without flipping 1D
    offs[0] = -2;
    expectedOut[0] = 0;
    expectedOut[1] = 0;
    expectedOut[2] = 0;
    expectedOut[3] = 1;
    expectedOut[4] = 2;
    expectedOut[5] = 3;
    expectedOut[6] = 4;
    flip[0] = false;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop partially within data range (right) without flipping 1D
    offs[0] = 10;
    expectedOut[0] = 10;
    expectedOut[1] = 11;
    expectedOut[2] = 12;
    expectedOut[3] = 13;
    expectedOut[4] = 0;
    expectedOut[5] = 0;
    expectedOut[6] = 0;
    flip[0] = false;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop partially within data range (left) without flipping 1D (mirror)
    offs[0] = -2;
    expectedOut[0] = 2;
    expectedOut[1] = 1;
    expectedOut[2] = 0;
    expectedOut[3] = 1;
    expectedOut[4] = 2;
    expectedOut[5] = 3;
    expectedOut[6] = 4;
    flip[0] = false;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, true);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop partially within data range (right) without flipping 1D (mirror)
    offs[0] = 10;
    expectedOut[0] = 10;
    expectedOut[1] = 11;
    expectedOut[2] = 12;
    expectedOut[3] = 13;
    expectedOut[4] = 12;
    expectedOut[5] = 11;
    expectedOut[6] = 10;
    flip[0] = false;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, true);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop out of data range (left) without flipping 1D
    offs[0] = -20;
    expectedOut[0] = 0;
    expectedOut[1] = 0;
    expectedOut[2] = 0;
    expectedOut[3] = 0;
    expectedOut[4] = 0;
    expectedOut[5] = 0;
    expectedOut[6] = 0;
    flip[0] = false;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop out of data range (right) without flipping 1D
    offs[0] = 50;
    expectedOut[0] = 0;
    expectedOut[1] = 0;
    expectedOut[2] = 0;
    expectedOut[3] = 0;
    expectedOut[4] = 0;
    expectedOut[5] = 0;
    expectedOut[6] = 0;
    flip[0] = false;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop out of data range (left) without flipping 1D (mirror)
    offs[0] = -30;
    expectedOut[0] = 4;
    expectedOut[1] = 3;
    expectedOut[2] = 2;
    expectedOut[3] = 1;
    expectedOut[4] = 0;
    expectedOut[5] = 1;
    expectedOut[6] = 2;
    flip[0] = false;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, true);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop out of data range (right) without flipping 1D (mirror)
    offs[0] = 50;
    expectedOut[0] = 2;
    expectedOut[1] = 1;
    expectedOut[2] = 0;
    expectedOut[3] = 1;
    expectedOut[4] = 2;
    expectedOut[5] = 3;
    expectedOut[6] = 4;
    flip[0] = false;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, true);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop out of data range (left) with flipping 1D
    offs[0] = -20;
    expectedOut[0] = 0;
    expectedOut[1] = 0;
    expectedOut[2] = 0;
    expectedOut[3] = 0;
    expectedOut[4] = 0;
    expectedOut[5] = 0;
    expectedOut[6] = 0;
    flip[0] = true;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop out of data range (right) with flipping 1D
    offs[0] = 50;
    expectedOut[0] = 0;
    expectedOut[1] = 0;
    expectedOut[2] = 0;
    expectedOut[3] = 0;
    expectedOut[4] = 0;
    expectedOut[5] = 0;
    expectedOut[6] = 0;
    flip[0] = true;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop out of data range (left) with flipping 1D (mirror)
    offs[0] = -30;
    expectedOut[0] = 2;
    expectedOut[1] = 1;
    expectedOut[2] = 0;
    expectedOut[3] = 1;
    expectedOut[4] = 2;
    expectedOut[5] = 3;
    expectedOut[6] = 4;
    flip[0] = true;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, true);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop out of data range (right) with flipping 1D (mirror)
    offs[0] = 50;
    expectedOut[0] = 4;
    expectedOut[1] = 3;
    expectedOut[2] = 2;
    expectedOut[3] = 1;
    expectedOut[4] = 0;
    expectedOut[5] = 1;
    expectedOut[6] = 2;
    flip[0] = true;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, true);
    EXPECT_EQ(toString(expectedOut), toString(out));
  }

  // 2D tests
  {
    // Crop within data range without flipping 2D
    std::vector<TypeParam> in(4 * 5);
    for (int i = 0; i < 4 * 5; ++i) in[i] = i;
    vector<int> inShape(2);
    inShape[0] = 4;
    inShape[1] = 5;
    vector<int> offs(2);
    offs[0] = 1;
    offs[1] = 2;
    vector<bool> flip(2, false);

    vector<TypeParam> expectedOut(2 * 3);
    expectedOut[0] = 7;
    expectedOut[1] = 8;
    expectedOut[2] = 9;
    expectedOut[3] = 12;
    expectedOut[4] = 13;
    expectedOut[5] = 14;

    vector<TypeParam> out(expectedOut.size(), 0);
    vector<int> outShape(2);
    outShape[0] = 2;
    outShape[1] = 3;

    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop within data range with flipping about dimension 0 2D
    expectedOut[0] = 12;
    expectedOut[1] = 13;
    expectedOut[2] = 14;
    expectedOut[3] = 7;
    expectedOut[4] = 8;
    expectedOut[5] = 9;
    flip[0] = true;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop within data range with flipping 2D
    expectedOut[0] = 14;
    expectedOut[1] = 13;
    expectedOut[2] = 12;
    expectedOut[3] = 9;
    expectedOut[4] = 8;
    expectedOut[5] = 7;
    flip[0] = true;
    flip[1] = true;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop within data range with flipping about dimension 1 2D
    expectedOut[0] = 9;
    expectedOut[1] = 8;
    expectedOut[2] = 7;
    expectedOut[3] = 14;
    expectedOut[4] = 13;
    expectedOut[5] = 12;
    flip[0] = false;
    flip[1] = true;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop partially within data range with flipping about dimension 1 2D
    offs[0] = 3;
    offs[1] = 4;
    expectedOut[0] = 0;
    expectedOut[1] = 0;
    expectedOut[2] = 19;
    expectedOut[3] = 0;
    expectedOut[4] = 0;
    expectedOut[5] = 0;
    flip[0] = false;
    flip[1] = true;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, false);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop partially within data range with flipping about dimension 1 2D
    // (mirror)
    offs[0] = 3;
    offs[1] = 4;
    expectedOut[0] = 17;
    expectedOut[1] = 18;
    expectedOut[2] = 19;
    expectedOut[3] = 12;
    expectedOut[4] = 13;
    expectedOut[5] = 14;
    flip[0] = false;
    flip[1] = true;
    cropAndFlip(
        in.data(), 1, 1, inShape, out.data(), outShape, offs, flip, true);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop partially within data range with flipping about dimension 0 2D
    // multi-channel (mirror)
    for (int i = 0; i < 20; ++i) in.push_back(i + 20);
    out.resize(12, 0);
    expectedOut.resize(12, 0);
    offs[0] = -1;
    offs[1] = 4;
    expectedOut[0] = 4;
    expectedOut[1] = 3;
    expectedOut[2] = 2;
    expectedOut[3] = 9;
    expectedOut[4] = 8;
    expectedOut[5] = 7;
    expectedOut[6] = 24;
    expectedOut[7] = 23;
    expectedOut[8] = 22;
    expectedOut[9] = 29;
    expectedOut[10] = 28;
    expectedOut[11] = 27;
    flip[0] = true;
    flip[1] = false;
    cropAndFlip(
        in.data(), 1, 2, inShape, out.data(), outShape, offs, flip, true);
    EXPECT_EQ(toString(expectedOut), toString(out));

    // Crop partially within data range with flipping about dimension 0 2D
    // multi-sample (mirror)
    for (int i = 0; i < 20; ++i) in.push_back(i + 20);
    out.resize(12, 0);
    expectedOut.resize(12, 0);
    offs[0] = -1;
    offs[1] = 4;
    expectedOut[0] = 4;
    expectedOut[1] = 3;
    expectedOut[2] = 2;
    expectedOut[3] = 9;
    expectedOut[4] = 8;
    expectedOut[5] = 7;
    expectedOut[6] = 24;
    expectedOut[7] = 23;
    expectedOut[8] = 22;
    expectedOut[9] = 29;
    expectedOut[10] = 28;
    expectedOut[11] = 27;
    flip[0] = true;
    flip[1] = false;
    cropAndFlip(
        in.data(), 2, 1, inShape, out.data(), outShape, offs, flip, true);
    EXPECT_EQ(toString(expectedOut), toString(out));
  }

}

}  // end namespace caffe

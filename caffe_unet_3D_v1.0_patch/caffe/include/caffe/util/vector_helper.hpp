#ifndef CAFFE_VECTOR_HELPERS_HPP_
#define CAFFE_VECTOR_HELPERS_HPP_

#include <vector>
#include <sstream>
#include <iomanip>
#include "caffe/util/math_functions.hpp"

namespace caffe {
using std::vector;

template< typename T>
std::string toString(vector<T> v) {
  std::ostringstream oss;
  for (int i = 0; i < v.size(); ++i) {
    if( i == 0) { oss << "("; } else { oss << ","; }
    oss << v[i];
  }
  oss << ")";
  return oss.str();
}

template< typename T>
std::string toString(vector<T> v, int precision) {
  std::ostringstream oss;
  oss << std::setprecision(precision);
  for (int i = 0; i < v.size(); ++i) {
    if( i == 0) { oss << "("; } else { oss << ","; }
    oss << v[i];
  }
  oss << ")";
  return oss.str();
}

inline
std::string toString(const int* v, int n) {
  std::ostringstream oss;
  for (int i = 0; i < n; ++i) {
    if (i == 0) { oss << "("; } else { oss << ","; }
    oss << v[i];
  }
  oss << ")";
  return oss.str();
}

inline
std::string toString(const float* v, int n) {
  std::ostringstream oss;
  for (int i = 0; i < n; ++i) {
    if (i == 0) { oss << "("; } else { oss << ","; }
    oss << v[i];
  }
  oss << ")";
  return oss.str();
}

inline
std::string toString(const double* v, int n) {
  std::ostringstream oss;
  for (int i = 0; i < n; ++i) {
    if (i == 0) { oss << "("; } else { oss << ","; }
    oss << v[i];
  }
  oss << ")";
  return oss.str();
}

template< typename T>
std::string Array2DtoString(
    const T* v, int ny, int nx, float precision = 1e-5) {
  std::ostringstream oss;
  int i = 0;
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      float u = v[i];
      u = floor(u / precision + 0.5) * precision;
      oss << u << ",";
      ++i;
    }
    oss << "\n";
  }
  return oss.str();
}

inline
vector<int> make_int_vect(int v0) {
  vector<int> vec(1);
  vec[0] = v0;
  return vec;
}

inline
vector<int> make_int_vect(int v0, int v1) {
  vector<int> vec(2);
  vec[0] = v0;
  vec[1] = v1;
  return vec;
}

inline
vector<int> make_int_vect(int v0, int v1, int v2) {
  vector<int> vec(3);
  vec[0] = v0;
  vec[1] = v1;
  vec[2] = v2;
  return vec;
}

inline
vector<int> make_int_vect(int v0, int v1, int v2, int v3) {
  vector<int> vec(4);
  vec[0] = v0;
  vec[1] = v1;
  vec[2] = v2;
  vec[3] = v3;
  return vec;
}

inline
vector<int> make_int_vect(int v0, int v1, int v2, int v3, int v4) {
  vector<int> vec(5);
  vec[0] = v0;
  vec[1] = v1;
  vec[2] = v2;
  vec[3] = v3;
  vec[4] = v4;
  return vec;
}

template<typename T>
inline
vector<T> make_vec( T v0) {
  vector<T> vec(1);
  vec[0] = v0;
  return vec;
}

template<typename T>
inline
vector<T> make_vec( T v0, T v1) {
  vector<T> vec(2);
  vec[0] = v0;
  vec[1] = v1;
  return vec;
}

template<typename T>
inline
vector<T> make_vec( T v0, T v1, T v2) {
  vector<T> vec(3);
  vec[0] = v0;
  vec[1] = v1;
  vec[2] = v2;
  return vec;
}

template<typename T>
inline
vector<T> make_vec( T v0, T v1, T v2, T v3) {
  vector<T> vec(4);
  vec[0] = v0;
  vec[1] = v1;
  vec[2] = v2;
  vec[3] = v3;
  return vec;
}

template<typename T>
inline
vector<T> make_vec( T v0, T v1, T v2, T v3, T v4) {
  vector<T> vec(5);
  vec[0] = v0;
  vec[1] = v1;
  vec[2] = v2;
  vec[3] = v3;
  vec[4] = v4;
  return vec;
}
template<typename T>
inline
vector<T> make_vec( T v0, T v1, T v2, T v3, T v4, T v5) {
  vector<T> vec(6);
  vec[0] = v0;
  vec[1] = v1;
  vec[2] = v2;
  vec[3] = v3;
  vec[4] = v4;
  vec[5] = v5;
  return vec;
}

template<typename T>
inline
vector<T> make_vec( T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8) {
  vector<T> vec(9);
  vec[0] = v0;
  vec[1] = v1;
  vec[2] = v2;
  vec[3] = v3;
  vec[4] = v4;
  vec[5] = v5;
  vec[6] = v6;
  vec[7] = v7;
  vec[8] = v8;
  return vec;
}

template<typename T>
inline
vector<T> make_vec( T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
                         T v8, T v9, T v10, T v11) {
  vector<T> vec(12);
  vec[0] = v0;
  vec[1] = v1;
  vec[2] = v2;
  vec[3] = v3;
  vec[4] = v4;
  vec[5] = v5;
  vec[6] = v6;
  vec[7] = v7;
  vec[8] = v8;
  vec[9] = v9;
  vec[10] = v10;
  vec[11] = v11;
  return vec;
}

template<typename T>
inline
vector<T> make_vec( T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
                         T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) {
  vector<T> vec(16);
  vec[0] = v0;
  vec[1] = v1;
  vec[2] = v2;
  vec[3] = v3;
  vec[4] = v4;
  vec[5] = v5;
  vec[6] = v6;
  vec[7] = v7;
  vec[8] = v8;
  vec[9] = v9;
  vec[10] = v10;
  vec[11] = v11;
  vec[12] = v12;
  vec[13] = v13;
  vec[14] = v14;
  vec[15] = v15;
  return vec;
}


template<typename T>
inline
vector<T> m_shift3D( T s1, T s2, T s3, const vector<T>& M)  {
  vector<T> A = make_vec<T>(1,0,0,s1,
                            0,1,0,s2,
                            0,0,1,s3,
                            0,0,0,1);
  vector<T> B(16);
  caffe_cpu_gemm<T>(CblasNoTrans, CblasNoTrans,
                    4, 4, 4, 1, A.data(), M.data(), 0, B.data());
  return B;
}

template<typename T>
inline
vector<T> m_scale3D( T s1, T s2, T s3, const vector<T>& M)  {
  vector<T> A = make_vec<T>(s1,0,0,0,
                            0,s2,0,0,
                            0,0,s3,0,
                            0,0,0, 1);
  vector<T> B(16);
  caffe_cpu_gemm<T>(CblasNoTrans, CblasNoTrans,
                    4, 4, 4, 1, A.data(), M.data(), 0, B.data());
  return B;
}

template<typename T>
inline
vector<T> m_rotate3D( T phi1, T phi2, T phi3, const vector<T>& M)  {
  T sina = sin( phi1 / 180 * M_PI);
  T cosa = cos( phi1 / 180 * M_PI);
  vector<T> rotateAroundZ = make_vec<T>(
      1,   0,     0,    0,
      0,   cosa, -sina, 0,
      0,   sina, cosa,  0,
      0,   0,     0,    1);


  sina = sin( phi2 / 180 * M_PI);
  cosa = cos( phi2 / 180 * M_PI);
  vector<T> rotateAroundY = make_vec<T>(
      cosa,  0,  -sina, 0,
      0,     1,   0,    0,
      sina,  0,   cosa, 0,
      0,     0,     0,  1);


  sina = sin( phi3 / 180 * M_PI);
  cosa = cos( phi3 / 180 * M_PI);
  vector<T> rotateAroundX = make_vec<T>(
      cosa, -sina, 0, 0,
      sina, cosa,  0, 0,
      0,    0,     1, 0,
      0,    0,     0, 1);

  vector<T> A(16);
  vector<T> B(16);
  caffe_cpu_gemm<T>(CblasNoTrans, CblasNoTrans, 4, 4, 4, 1,
                    rotateAroundZ.data(), M.data(), 0, A.data());
  caffe_cpu_gemm<T>(CblasNoTrans, CblasNoTrans, 4, 4, 4, 1,
                    rotateAroundY.data(), A.data(), 0, B.data());
  caffe_cpu_gemm<T>(CblasNoTrans, CblasNoTrans, 4, 4, 4, 1,
                    rotateAroundX.data(), B.data(), 0, A.data());
  return A;
}

// Example for N=5
// index: -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9 10 11 12
// value:  0  1  2  3  4  3  2  1  0  1  2  3  4  3  2  1  0  1  2  3  4
inline
int mirrorAtBorder( int i, int N) {
  if( i >= 0 && i < N) return i;
  if( N == 1) return 0;
  int i2 = abs(i) % (N*2-2);
  if( i2 >= N) {
    i2 = (N*2-2) - i2;
  }
  return i2;
}

template< typename T>
inline
T linear2D_zeropad( const T* data, int ny, int nx, float y, float x) {
  const int iy = floor(y);
  const int ix = floor(x);
  const float fy = y - iy;
  const float fx = x - ix;

  const bool validx0 = (ix >= 0 && ix < nx);
  const bool validy0 = (iy >= 0 && iy < ny);
  const bool validx1 = (ix+1 >= 0 && ix+1 < nx);
  const bool validy1 = (iy+1 >= 0 && iy+1 < ny);

  const T* p = data + iy * nx + ix;
  const float a00 = (validy0 && validx0)? p[     0] : 0;
  const float a01 = (validy0 && validx1)? p[     1] : 0;
  const float a10 = (validy1 && validx0)? p[nx    ] : 0;
  const float a11 = (validy1 && validx1)? p[nx + 1] : 0;

  return (1 - fy) * (1 - fx) * a00
      +  (1 - fy) * (    fx) * a01
      +  (    fy) * (1 - fx) * a10
      +  (    fy) * (    fx) * a11;
}


template< typename T>
inline
T linear2D_mirror( const T* data, int ny, int nx, float y, float x) {
  const int iy = floor(y);
  const int ix = floor(x);
  const float fy = y - iy;
  const float fx = x - ix;

  const int iy0 = mirrorAtBorder( iy    , ny);
  const int ix0 = mirrorAtBorder( ix    , nx);
  const int iy1 = mirrorAtBorder( iy + 1, ny);
  const int ix1 = mirrorAtBorder( ix + 1, nx);

  return (1 - fy) * (1 - fx) * data[iy0 * nx + ix0]
      +  (1 - fy) * (    fx) * data[iy0 * nx + ix1]
      +  (    fy) * (1 - fx) * data[iy1 * nx + ix0]
      +  (    fy) * (    fx) * data[iy1 * nx + ix1];
}


template< typename T>
inline
T linear3D_zeropad( const T* data, int nz, int ny, int nx,
                     float z, float y, float x) {
  const int iz = floor(z);
  const int iy = floor(y);
  const int ix = floor(x);
  const float fz = z - iz;
  const float fy = y - iy;
  const float fx = x - ix;


  const bool validz0 = (iz >= 0 && iz < nz);
  const bool validy0 = (iy >= 0 && iy < ny);
  const bool validx0 = (ix >= 0 && ix < nx);
  const bool validz1 = (iz+1 >= 0 && iz+1 < nz);
  const bool validy1 = (iy+1 >= 0 && iy+1 < ny);
  const bool validx1 = (ix+1 >= 0 && ix+1 < nx);


  int nxy = nx * ny;
  const T* p = data + iz * nxy + iy * nx + ix;
  const float a000 = (validz0 && validy0 && validx0)? p[  0 +  0 + 0] : 0;
  const float a001 = (validz0 && validy0 && validx1)? p[  0 +  0 + 1] : 0;
  const float a010 = (validz0 && validy1 && validx0)? p[  0 + nx + 0] : 0;
  const float a011 = (validz0 && validy1 && validx1)? p[  0 + nx + 1] : 0;
  const float a100 = (validz1 && validy0 && validx0)? p[nxy +  0 + 0] : 0;
  const float a101 = (validz1 && validy0 && validx1)? p[nxy +  0 + 1] : 0;
  const float a110 = (validz1 && validy1 && validx0)? p[nxy + nx + 0] : 0;
  const float a111 = (validz1 && validy1 && validx1)? p[nxy + nx + 1] : 0;

  return (1 - fz) * (1 - fy) * (1 - fx) * a000
      +  (1 - fz) * (1 - fy) * (    fx) * a001
      +  (1 - fz) * (    fy) * (1 - fx) * a010
      +  (1 - fz) * (    fy) * (    fx) * a011
      +  (    fz) * (1 - fy) * (1 - fx) * a100
      +  (    fz) * (1 - fy) * (    fx) * a101
      +  (    fz) * (    fy) * (1 - fx) * a110
      +  (    fz) * (    fy) * (    fx) * a111;
}


template< typename T>
inline
T linear3D_mirror( const T* data, int nz, int ny, int nx,
                     float z, float y, float x) {
  const int iz = floor(z);
  const int iy = floor(y);
  const int ix = floor(x);
  const float fz = z - iz;
  const float fy = y - iy;
  const float fx = x - ix;

  const int iz0 = mirrorAtBorder( iz    , nz);
  const int iy0 = mirrorAtBorder( iy    , ny);
  const int ix0 = mirrorAtBorder( ix    , nx);
  const int iz1 = mirrorAtBorder( iz + 1, nz);
  const int iy1 = mirrorAtBorder( iy + 1, ny);
  const int ix1 = mirrorAtBorder( ix + 1, nx);

  return (1 - fz) * (1 - fy) * (1 - fx) * data[(iz0 * ny + iy0) * nx + ix0]
      +  (1 - fz) * (1 - fy) * (    fx) * data[(iz0 * ny + iy0) * nx + ix1]
      +  (1 - fz) * (    fy) * (1 - fx) * data[(iz0 * ny + iy1) * nx + ix0]
      +  (1 - fz) * (    fy) * (    fx) * data[(iz0 * ny + iy1) * nx + ix1]
      +  (    fz) * (1 - fy) * (1 - fx) * data[(iz1 * ny + iy0) * nx + ix0]
      +  (    fz) * (1 - fy) * (    fx) * data[(iz1 * ny + iy0) * nx + ix1]
      +  (    fz) * (    fy) * (1 - fx) * data[(iz1 * ny + iy1) * nx + ix0]
      +  (    fz) * (    fy) * (    fx) * data[(iz1 * ny + iy1) * nx + ix1];
}

template< typename T>
inline
T nearest2D_zeropad( const T* data, int ny, int nx, float y, float x) {
  const int iy = floor(y + 0.5);
  const int ix = floor(x + 0.5);
  return (ix >= 0 && ix < nx
	  && iy >= 0 && iy < ny)?  data[ iy * nx + ix] : 0;
}

template< typename T>
inline
T nearest2D_mirror( const T* data, int ny, int nx, float y, float x) {
  const int iy = mirrorAtBorder(floor(y + 0.5), ny);
  const int ix = mirrorAtBorder(floor(x + 0.5), nx);
  return data[ iy * nx + ix];
}

template< typename T>
inline
T nearest3D_zeropad( const T* data, int nz, int ny, int nx,
             float z, float y, float x) {
  const int iz = floor(z + 0.5);
  const int iy = floor(y + 0.5);
  const int ix = floor(x + 0.5);
  return (ix >= 0 && ix < nx
	  && iy >= 0 && iy < ny
	  && iz >= 0 && iz < nz)?
    data[ iz * ny * nx + iy * nx + ix] : 0;
}

template< typename T>
inline
T nearest3D_mirror( const T* data, int nz, int ny, int nx,
             float z, float y, float x) {
  const int iz = mirrorAtBorder(floor(z + 0.5), nz);
  const int iy = mirrorAtBorder(floor(y + 0.5), ny);
  const int ix = mirrorAtBorder(floor(x + 0.5), nx);
  return data[ iz * ny * nx + iy * nx + ix];
}


template< typename T, typename Functor>
inline
void transform2D(const T* in, int inNb,  int inNc,  int inNz,  int inNy,  int inNx,
                 const T* def,int defNb,            int defNz, int defNy, int defNx,
                 T* out,      int outNb, int outNc, int outNz, int outNy, int outNx,
                 Functor Interpolator) {
  CHECK_EQ(inNb, defNb) << "in and deform must have same batch size";
  CHECK_EQ(inNb, outNb) << "in and out must have same batch size";
  CHECK_EQ(inNc, outNc) << "in and out must have same number of channels";
  CHECK_GE(defNz, outNz);
  CHECK_GE(defNy, outNy);
  CHECK_GE(defNx, outNx);
  CHECK_EQ((defNz - outNz) % 2, 0);
  CHECK_EQ((defNy - outNy) % 2, 0);
  CHECK_EQ((defNx - outNx) % 2, 0);

  const int offsZ = (defNz - outNz) / 2;
  const int offsY = (defNy - outNy) / 2;
  const int offsX = (defNx - outNx) / 2;

  for (int n = 0; n < outNb; ++n) {
    for (int c = 0; c < outNc; ++c) {
      for (int z = 0; z < outNz; ++z) {
        const T* inSlice  = in  + ((n * inNc  + c) * inNz  + z) * inNy  * inNx;
        const T* defSlice = def + ( n * defNz + z + offsZ) * defNy * defNx * 2;
        T*       outP     = out + ((n * outNc + c) * outNz + z) * outNy * outNx;
        for (int y = 0; y < outNy; ++y) {
          const T* defP = defSlice + ((y + offsY) * defNx + offsX) * 2;
          for( int x = 0; x < outNx; ++x) {
            *outP = Interpolator( inSlice, inNy, inNx, defP[0], defP[1]);
            ++outP;
            defP += 2;
          }
        }
      }
    }
  }
}


template< typename T, typename Functor>
inline
void transform2D_offset(const T* in, int inNb,  int inNc,  int inNz,  int inNy,  int inNx,
                 const T* def,int defNb,            int defNz, int defNy, int defNx,
				 int offsetY, int offsetX,
                 T* out,      int outNb, int outNc, int outNz, int outNy, int outNx,
                 Functor Interpolator) {
  CHECK_EQ(inNb, defNb) << "in and deform must have same batch size";
  CHECK_EQ(inNb, outNb) << "in and out must have same batch size";
  CHECK_EQ(inNc, outNc) << "in and out must have same number of channels";
  CHECK_LE(defNz, outNz);
  CHECK_LE(defNy, outNy);
  CHECK_LE(defNx, outNx);
  CHECK_EQ((defNz - outNz) % 2, 0);
  CHECK_EQ((defNy - outNy) % 2, 0);
  CHECK_EQ((defNx - outNx) % 2, 0);

  const int offsZ = (defNz - outNz) / 2;
  const int offsY = (defNy - outNy) / 2;
  const int offsX = (defNx - outNx) / 2;

  for (int n = 0; n < outNb; ++n) {
    for (int c = 0; c < outNc; ++c) {
      for (int z = 0; z < outNz; ++z) {
        const T* inSlice  = in  + ((n * inNc  + c) * inNz  + z) * inNy  * inNx;
        const T* defSlice = def + ( n * defNz + z + offsZ) * defNy * defNx * 2;
        T*       outP     = out + ((n * outNc + c) * outNz + z) * outNy * outNx;
        for (int y = 0; y < outNy; ++y) {
          const T* defP = defSlice + ((y + offsY) * defNx + offsX) * 2;
          for( int x = 0; x < outNx; ++x) {
            *outP = Interpolator( inSlice, inNy, inNx, defP[0]+offsetY, defP[1]+offsetX);
            ++outP;
            defP += 2;
          }
        }
      }
    }
  }
}


template< typename T>
inline
void flip2D_x(const T* in, int inNb,  int inNc,  int inNz,  int inNy,  int inNx,
			  T* out,      int outNb, int outNc, int outNz, int outNy, int outNx) {
  CHECK_EQ(inNb, outNb) << "in and out must have same batch size";
  CHECK_EQ(inNc, outNc) << "in and out must have same number of channels";
  CHECK_EQ(inNz, outNz) << "in and out must have same number of elements in z";
  CHECK_EQ(inNy, outNy) << "in and out must have same number of elements in y";
  CHECK_EQ(inNx, outNx) << "in and out must have same number of elements in x";

  for (int n = 0; n < outNb; ++n) {
    for (int c = 0; c < outNc; ++c) {
      for (int z = 0; z < outNz; ++z) {
        const T* inSlice  = in  + ((n * inNc  + c) * inNz  + z) * inNy  * inNx;
        T*       outP     = out + ((n * outNc + c) * outNz + z) * outNy * outNx;
        for (int y = 0; y < outNy; ++y) {
		  const T* inP = inSlice + (y * inNx) + inNx - 1;
          for( int x = 0; x < outNx; ++x) {
            *outP = *inP;
            ++outP;
			--inP;
          }
        }
      }
    }
  }
}


template< typename T>
inline
void flip2D_y(const T* in, int inNb,  int inNc,  int inNz,  int inNy,  int inNx,
			  T* out,      int outNb, int outNc, int outNz, int outNy, int outNx) {
  CHECK_EQ(inNb, outNb) << "in and out must have same batch size";
  CHECK_EQ(inNc, outNc) << "in and out must have same number of channels";
  CHECK_EQ(inNz, outNz) << "in and out must have same number of elements in z";
  CHECK_EQ(inNy, outNy) << "in and out must have same number of elements in y";
  CHECK_EQ(inNx, outNx) << "in and out must have same number of elements in x";

  for (int n = 0; n < outNb; ++n) {
    for (int c = 0; c < outNc; ++c) {
      for (int z = 0; z < outNz; ++z) {
        const T* inSlice  = in  + ((n * inNc  + c) * inNz  + z) * inNy  * inNx;
        T*       outP     = out + ((n * outNc + c) * outNz + z) * outNy * outNx;
        for (int y = 0; y < outNy; ++y) {
		  const T* inP = inSlice + ((outNy - y - 1) * inNx);
          for( int x = 0; x < outNx; ++x) {
            *outP = *inP;
            ++outP;
			++inP;
          }
        }
      }
    }
  }
}


template< typename T>
inline
void flip2D_xy(const T* in, int inNb,  int inNc,  int inNz,  int inNy,  int inNx,
			   T* out,      int outNb, int outNc, int outNz, int outNy, int outNx) {
  CHECK_EQ(inNb, outNb) << "in and out must have same batch size";
  CHECK_EQ(inNc, outNc) << "in and out must have same number of channels";
  CHECK_EQ(inNz, outNz) << "in and out must have same number of elements in z";
  CHECK_EQ(inNy, outNy) << "in and out must have same number of elements in y";
  CHECK_EQ(inNx, outNx) << "in and out must have same number of elements in x";

  for (int n = 0; n < outNb; ++n) {
    for (int c = 0; c < outNc; ++c) {
      for (int z = 0; z < outNz; ++z) {
        const T* inSlice  = in  + ((n * inNc  + c) * inNz  + z) * inNy  * inNx;
        T*       outP     = out + ((n * outNc + c) * outNz + z) * outNy * outNx;
        for (int y = 0; y < outNy; ++y) {
		  const T* inP = inSlice + ((outNy - y - 1) * inNx) + inNx - 1;
          for( int x = 0; x < outNx; ++x) {
            *outP = *inP;
            ++outP;
			--inP;
          }
        }
      }
    }
  }
}


template< typename T, typename Functor>
inline
void transform3D(const T* in, int inNb,  int inNc,  int inNz,  int inNy,  int inNx,
                 const T* def,int defNb,            int defNz, int defNy, int defNx,
                 T* out,      int outNb, int outNc, int outNz, int outNy, int outNx,
                 Functor Interpolator) {
  CHECK_EQ(inNb, defNb) << "in and deform must have same batch size";
  CHECK_EQ(inNb, outNb) << "in and out must have same batch size";
  CHECK_EQ(inNc, outNc) << "in and out must have same number of channels";
  CHECK_GE(defNz, outNz);
  CHECK_GE(defNy, outNy);
  CHECK_GE(defNx, outNx);
  CHECK_EQ((defNz - outNz) % 2, 0);
  CHECK_EQ((defNy - outNy) % 2, 0);
  CHECK_EQ((defNx - outNx) % 2, 0);

  const int offsZ = (defNz - outNz) / 2;
  const int offsY = (defNy - outNy) / 2;
  const int offsX = (defNx - outNx) / 2;

  for (int n = 0; n < outNb; ++n) {
    for (int c = 0; c < outNc; ++c) {
      const T* inBlock  = in  + (n * inNc  + c) * inNz * inNy  * inNx;
      const T* defBlock = def + (n * defNz + offsZ) * defNy * defNx * 3;
      T*       outP     = out + (n * outNc + c) * outNz * outNy * outNx;
      for (int z = 0; z < outNz; ++z) {
        const T* defSlice = defBlock + z * defNy * defNx * 3;
        for (int y = 0; y < outNy; ++y) {
          const T* defP = defSlice + ((y + offsY) * defNx + offsX) * 3;
          for( int x = 0; x < outNx; ++x) {
            *outP = Interpolator( inBlock, inNz, inNy, inNx, defP[0], defP[1], defP[2]);
            ++outP;
            defP += 3;
          }
        }
      }
    }
  }
}

template< typename T, typename Functor>
inline
void transform3D_offset(
    const T* in, int inNb,  int inNc,  int inNz,  int inNy,  int inNx,
    const T* def,int defNb,            int defNz, int defNy, int defNx,
                                       int dZ,    int dY,    int dX,
    T* out,      int outNb, int outNc, int outNz, int outNy, int outNx,
    Functor Interpolator) {
  CHECK_EQ(inNb, defNb) << "in and deform must have same batch size";
  CHECK_EQ(inNb, outNb) << "in and out must have same batch size";
  CHECK_EQ(inNc, outNc) << "in and out must have same number of channels";
  CHECK_GE(defNz, outNz);
  CHECK_GE(defNy, outNy);
  CHECK_GE(defNx, outNx);
  CHECK_EQ((defNz - outNz) % 2, 0);
  CHECK_EQ((defNy - outNy) % 2, 0);
  CHECK_EQ((defNx - outNx) % 2, 0);

  const int offsZ = (defNz - outNz) / 2;
  const int offsY = (defNy - outNy) / 2;
  const int offsX = (defNx - outNx) / 2;

  for (int n = 0; n < outNb; ++n) {
    for (int c = 0; c < outNc; ++c) {
      const T* inBlock  = in  + (n * inNc  + c) * inNz * inNy  * inNx;
      const T* defBlock = def + (n * defNz + offsZ) * defNy * defNx * 3;
      T*       outP     = out + (n * outNc + c) * outNz * outNy * outNx;
      for (int z = 0; z < outNz; ++z) {
        const T* defSlice = defBlock + z * defNy * defNx * 3;
        for (int y = 0; y < outNy; ++y) {
          const T* defP = defSlice + ((y + offsY) * defNx + offsX) * 3;
          for( int x = 0; x < outNx; ++x) {
            *outP = Interpolator(
                inBlock, inNz, inNy, inNx,
                defP[0] + dZ, defP[1] + dY, defP[2] + dX);
            ++outP;
            defP += 3;
          }
        }
      }
    }
  }
}

  template<typename TOut, typename TIn>
  std::vector<TOut> vector_cast(std::vector<TIn> const &vec)
  {
    std::vector<TOut> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        result[i] = static_cast<TOut>(vec[i]);
    return result;
  }

  template<typename T>
  std::vector<T> floor(std::vector<T> const &vec)
  {
    std::vector<T> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) result[i] = std::floor(vec[i]);
    return result;
  }

  template<typename T>
  std::vector<T> round(std::vector<T> const &vec)
  {
    std::vector<T> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        result[i] = std::floor(vec[i] + 0.5);
    return result;
  }

  template<typename T>
  std::vector<T> ceil(std::vector<T> const &vec)
  {
    std::vector<T> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) result[i] = std::ceil(vec[i]);
    return result;
  }

  template<typename T>
  T sum(std::vector<T> const &vec)
  {
    T result = 0;
    for (size_t i = 0; i < vec.size(); ++i) result += vec[i];
    return result;
  }

  template<typename T>
  T product(std::vector<T> const &vec)
  {
    T result = 1;
    for (size_t i = 0; i < vec.size(); ++i) result *= vec[i];
    return result;
  }

  template<typename T>
  std::vector<T> operator+(std::vector<T> const &a, std::vector<T> const &b) {
    CHECK_EQ(a.size(), b.size()) << "Cannot add vectors of different lengths";
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
    return result;
  }

  template<typename T>
  std::vector<T> operator+(std::vector<T> const &a, T const &b) {
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b;
    return result;
  }

  template<typename T>
  std::vector<T> operator+(T const &a, std::vector<T> const &b) {
    std::vector<T> result(b.size());
    for (size_t i = 0; i < b.size(); ++i) result[i] = a + b[i];
    return result;
  }

  template<typename T>
  std::vector<T> operator-(std::vector<T> const &a, std::vector<T> const &b) {
    CHECK_EQ(a.size(), b.size())
        << "Cannot compute difference of vectors of different lengths";
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b[i];
    return result;
  }

  template<typename T>
  std::vector<T> operator-(std::vector<T> const &a, T const &b) {
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b;
    return result;
  }

  template<typename T>
  std::vector<T> operator-(T const &a, std::vector<T> const &b) {
    std::vector<T> result(b.size());
    for (size_t i = 0; i < b.size(); ++i) result[i] = a - b[i];
    return result;
  }

  template<typename T>
  std::vector<T> operator*(std::vector<T> const &a, std::vector<T> const &b) {
    CHECK_EQ(a.size(), b.size())
        << "Cannot apply elementwise multiplication to vectors of different "
        << "lengths";
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] * b[i];
    return result;
  }

  template<typename T>
  T dot(std::vector<T> const &a, std::vector<T> const &b) {
    CHECK_EQ(a.size(), b.size())
        << "Cannot apply elementwise multiplication to vectors of different "
        << "lengths";
    T result = 1;
    for (size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
  }

  template<typename T>
  std::vector<T> operator*(std::vector<T> const &a, T const &b) {
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] * b;
    return result;
  }

  template<typename T>
  std::vector<T> operator*(T const &a, std::vector<T> const &b) {
    std::vector<T> result(b.size());
    for (size_t i = 0; i < b.size(); ++i) result[i] = a * b[i];
    return result;
  }

  template<typename T>
  std::vector<T> operator/(std::vector<T> const &a, std::vector<T> const &b) {
    CHECK_EQ(a.size(), b.size())
        << "Cannot apply elementwise division to vectors of different "
        << "lengths";
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] / b[i];
    return result;
  }

  template<typename T>
  std::vector<T> operator/(std::vector<T> const &a, T const &b) {
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] / b;
    return result;
  }

  template<typename T>
  std::vector<T> operator/(T const &a, std::vector<T> const &b) {
    std::vector<T> result(b.size());
    for (size_t i = 0; i < b.size(); ++i) result[i] = a / b[i];
    return result;
  }

  template<typename T>
  inline
  void cropAndFlip(
      const T* in, int nSamples, int nChannels, std::vector<int> const &inShape,
      T *out, std::vector<int> const &outShape,
      std::vector<int> const &offset, std::vector<bool> const &flip,
      bool padMirror) {
    CHECK_EQ(inShape.size(), outShape.size())
        << "in and out must have same dimensionality";
    CHECK_EQ(offset.size(), outShape.size())
        << "offset and out must have same dimensionality";
    CHECK_EQ(flip.size(), outShape.size())
        << "flip and out must have same dimensionality";

    int nDims = inShape.size();
    int outSize = product(outShape);
    std::vector<int> inStrides(nDims, 1);
    for (int d = nDims - 2; d >= 0; --d)
        inStrides[d] = inStrides[d + 1] * inShape[d + 1];
    int inSize = inStrides[0] * inShape[0];

    for (int n = 0; n < nSamples; ++n) {
      for (int c = 0; c < nChannels; ++c) {
        const T* inBlock  = in  + (n * nChannels + c) * inSize;
        T*       outP     = out + (n * nChannels + c) * outSize;
        for (int i = 0; i < outSize; ++i, ++outP) {
          const T* inP = inBlock;
          int tmp = i;
          bool skip = false;
          for (int d = nDims - 1; d >= 0 && !skip; --d) {
            int src = offset[d] +
                ((flip[d]) ?
                 (outShape[d] - (tmp % outShape[d]) - 1) : (tmp % outShape[d]));
            if (src < 0 || src >= inShape[d]) {
              if (padMirror) {
                if (src < 0) src = -src;
                int n = src / (inShape[d] - 1);
                if (n % 2 == 0) src = src - n * (inShape[d] - 1);
                else src = (n + 1) * (inShape[d] - 1) - src;
              }
              else {
                *outP = 0;
                skip = true;
              }
            }
            tmp /= outShape[d];
            inP += src * inStrides[d];
          }
          if (!skip) *outP = *inP;
        }
      }
    }
  }

}  // namespace caffe

#endif  // CAFFE_VECTOR_HELPERS_HPP_

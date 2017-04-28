#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<unsigned int>(const int N, const unsigned int alpha, unsigned int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}
template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);


template <typename Dtype>
void caffe_copy_subarray(const Dtype* src_p, const vector<int>& src_shape,
                         Dtype*       trg_p, const vector<int>& trg_shape,
                         const vector<int>& src_offset,
                         const vector<int>& copy_shape,
                         const vector<int>& trg_offset) {
//  std::cout << "copy_subarray called with \n"
//            << "src_shape=" << toString( src_shape) << "\n"
//            << "trg_shape=" << toString( trg_shape) << "\n"
//            << "src_offset=" << toString( src_offset) << "\n"
//            << "copy_shape=" << toString( copy_shape) << "\n"
//            << "trg_offset=" << toString( trg_offset) << "\n";
  int num_axes = src_shape.size();
  CHECK_LT(num_axes, 10) << "only 10 axes are supported";
  CHECK_EQ(src_shape.size(), trg_shape.size()) << "target must have same number of axes";
  CHECK_EQ(src_shape.size(), src_offset.size());
  CHECK_EQ(src_shape.size(), copy_shape.size());
  CHECK_EQ(src_shape.size(), trg_offset.size());

  int N[10];   // copy shape
  int so[10];  // src offset
  int to[10];  // trg offset
  int ss[10];  // src stride
  int ts[10];  // trg stride
  int axes_offset = 10 - num_axes;
  for (int i = 0; i < axes_offset; ++i) {
    N[i] = 1;
    so[i] = 0;
    to[i] = 0;
    ss[i] = 0;  // dummy value, will be multiplied with zero anyway
    ts[i] = 0;  // dummy value, will be multiplied with zero anyway
  }

  for (int i = 0; i < num_axes; ++i) {
    N[ axes_offset + i] = copy_shape[i];
    so[axes_offset + i] = src_offset[i];
    to[axes_offset + i] = trg_offset[i];
  }

  ss[9] = 1;
  ts[9] = 1;
  for (int i = num_axes-1; i > 0; --i) {
    ss[axes_offset + i - 1] = src_shape[i] * ss[axes_offset + i];
    ts[axes_offset + i - 1] = trg_shape[i] * ts[axes_offset + i];
  }
//  std::cout << "resulting 10d vectors\n"
//            << "N=" << toString(N,10) << "\n"
//            << "so=" << toString(so,10) << "\n"
//            << "to=" << toString(to,10) << "\n"
//            << "ss=" << toString(ss,10) << "\n"
//            << "ts=" << toString(ts,10) << "\n";

  int copy_nelem = N[9];
  for (                int i0 = 0; i0 < N[0]; ++i0) { int s0  =      ss[0] * (i0 + so[0]); int t0 =      ts[0] * (i0 + to[0]);
    for (              int i1 = 0; i1 < N[1]; ++i1) { int s1  = s0 + ss[1] * (i1 + so[1]); int t1 = t0 + ts[1] * (i1 + to[1]);
      for (            int i2 = 0; i2 < N[2]; ++i2) { int s2  = s1 + ss[2] * (i2 + so[2]); int t2 = t1 + ts[2] * (i2 + to[2]);
        for (          int i3 = 0; i3 < N[3]; ++i3) { int s3  = s2 + ss[3] * (i3 + so[3]); int t3 = t2 + ts[3] * (i3 + to[3]);
          for (        int i4 = 0; i4 < N[4]; ++i4) { int s4  = s3 + ss[4] * (i4 + so[4]); int t4 = t3 + ts[4] * (i4 + to[4]);
            for (      int i5 = 0; i5 < N[5]; ++i5) { int s5  = s4 + ss[5] * (i5 + so[5]); int t5 = t4 + ts[5] * (i5 + to[5]);
              for (    int i6 = 0; i6 < N[6]; ++i6) { int s6  = s5 + ss[6] * (i6 + so[6]); int t6 = t5 + ts[6] * (i6 + to[6]);
                for (  int i7 = 0; i7 < N[7]; ++i7) { int s7  = s6 + ss[7] * (i7 + so[7]); int t7 = t6 + ts[7] * (i7 + to[7]);
                  for (int i8 = 0; i8 < N[8]; ++i8) { int s8  = s7 + ss[8] * (i8 + so[8]); int t8 = t7 + ts[8] * (i8 + to[8]);
                    caffe_copy(copy_nelem, src_p + s8 + so[9], trg_p + t8 + to[9]);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template void caffe_copy_subarray<int>(         const int*          src_p, const vector<int>& src_shape, int*          trg_p, const vector<int>& trg_shape, const vector<int>& src_offset, const vector<int>& copy_shape, const vector<int>& trg_offset);
template void caffe_copy_subarray<unsigned int>(const unsigned int* src_p, const vector<int>& src_shape, unsigned int* trg_p, const vector<int>& trg_shape, const vector<int>& src_offset, const vector<int>& copy_shape, const vector<int>& trg_offset);
template void caffe_copy_subarray<float>(       const float*        src_p, const vector<int>& src_shape, float*        trg_p, const vector<int>& trg_shape, const vector<int>& src_offset, const vector<int>& copy_shape, const vector<int>& trg_offset);
template void caffe_copy_subarray<double>(      const double*       src_p, const vector<int>& src_shape, double*       trg_p, const vector<int>& trg_shape, const vector<int>& src_offset, const vector<int>& copy_shape, const vector<int>& trg_offset);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <typename Dtype>
size_t caffe_randi_arbitrary_cdf(const size_t n, const Dtype* cdf) {
  CHECK_GT(n, 0);
  CHECK(cdf);
  Dtype maxValue = cdf[n-1];
  Dtype r;
  caffe_rng_uniform( 1, Dtype(0), maxValue, &r);
  const Dtype* p = std::lower_bound( cdf, cdf + n, r);
  return p - cdf;
}

template
size_t caffe_randi_arbitrary_cdf(const size_t n, const float* cdf);

template
size_t caffe_randi_arbitrary_cdf(const size_t n, const double* cdf);

template <typename Dtype>
void caffe_rand_pos_arbitrary_cdf(const Dtype* cdf, int nz, int ny, int nx,
                                  int* z, int* y, int* x) {
  // sample a random index and map it to coordinates
  size_t idx = caffe_randi_arbitrary_cdf( nz * ny * nx, cdf);
  *z = idx / (ny * nx);
  idx = idx - *z * ny * nx;
  *y = idx / nx;
  *x = idx - *y * nx;
}

template
void caffe_rand_pos_arbitrary_cdf(const float* cdf, int nz, int ny, int nx,
                                  int* z, int* y, int* x);
template
void caffe_rand_pos_arbitrary_cdf(const double* cdf, int nz, int ny, int nx,
                                  int* z, int* y, int* x);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <typename Dtype>
void caffe_cpu_cumsum(const size_t n, const Dtype* x, Dtype* y) {
  CHECK_GE(n, 0);
  CHECK(x);
  CHECK(y);
  if( n == 0) return;
  Dtype cumsum = 0;
  for (size_t i = 0; i < n; ++i) {
    cumsum += x[i];
    y[i] = cumsum;
  }
}

template
void caffe_cpu_cumsum(const size_t n, const float* x, float* y);

template
void caffe_cpu_cumsum(const size_t n, const double* x, double* y);

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

}  // namespace caffe

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/value_augmentation_layer.hpp"

#include <stdexcept>

namespace caffe {

template<typename Dtype>
Dtype random_factor( Dtype max_factor) {
  CHECK_GE(max_factor, 1);
  if (max_factor == 1) return 1;
  Dtype sigma = log(max_factor)/3;
  Dtype v = 0;
  while( v < 1/max_factor || v > max_factor) {
    caffe_rng_gaussian<Dtype>( 1, 0, sigma, &v);
    v = exp(v);
  }
  return v;
}

template<typename Dtype>
Dtype uniform_random_value( Dtype minvalue, Dtype maxvalue) {
  Dtype v = 0;
  caffe_rng_uniform<Dtype>( 1, minvalue, maxvalue, &v);
  return v;
}


template <typename Dtype>
void ValueAugmentationLayer<Dtype>::CreateLinearInterpExtrapMatrix(
    int n_in,  Dtype dx_in,
    int n_out, Dtype dx_out,
    int n_extrapol,
    Dtype* lin_mat) {
  const int ncols = n_in;
  const int nrows = n_out + 2 * n_extrapol;

  // left side linear extrapolation
  // f(x) = y0 + x * (y1 - y0) / dx_in
  //      = (1 - x / dx_in) * y0   +   (x / dx_in) * y1
  //
  for (int row = 0; row < n_extrapol; ++row) {
    Dtype x_out = (row - n_extrapol) * dx_out;
    lin_mat[ row * ncols + 0] = 1 - x_out / dx_in;
    lin_mat[ row * ncols + 1] = x_out / dx_in;
    for (int col = 2; col < ncols; ++col) {
      lin_mat[ row * ncols + col] = 0;
    }
  }

  // interpolation
  // simple triangular kernel
  //
  for (int row = n_extrapol; row < n_out + n_extrapol; ++row) {
    Dtype x_out = (row - n_extrapol) * dx_out;
    for (int col = 0; col < ncols; ++col) {
      Dtype x_in = col * dx_in;
      lin_mat[ row * ncols + col] =
          std::max( Dtype(0), 1 - std::abs( x_out - x_in) / dx_in);
    }
  }

  // right side linear extrapolation
  // e.g. for 3 data points
  // f(x) = y2 + (x - x2) * (y2 - y1) / dx_in
  //      = (- (x - x2) / dx_in) * y1   +   (1 + (x - x2) / dx_in) * y2
  //
  for (int row = n_extrapol + n_out; row < nrows; ++row) {
    Dtype x_out = (row - (n_extrapol + n_out - 1)) * dx_out;
    for (int col = 0; col < ncols-2; ++col) {
      lin_mat[ row * ncols + col] = 0;
    }
    lin_mat[ row * ncols + ncols - 2] = - x_out / dx_in;
    lin_mat[ row * ncols + ncols - 1] = 1 + x_out / dx_in;
  }
}



template <typename Dtype>
void ValueAugmentationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  const ValueAugmentationParameter& param =
      this->layer_param_.value_augmentation_param();

  // check that arguments are valid
  CHECK_GE(param.lut_size(), 2)
      << "Lookup table must have at least two elements.";
  CHECK_GE(param.slope_max(), param.slope_min())
      << "Minimum slope must be smaller or equal than the maximum slope.";
  CHECK_GE(param.n_control_point_insertions(),0) << "Parameter must be positive";
  CHECK_LE(param.black_from(), param.black_to());
  CHECK_LE(param.white_from(), param.white_to());
  CHECK_LE(param.black_to(), param.white_from());

  // set parameters
  slope_min_        = param.slope_min();
  slope_max_        = param.slope_max();
  black_from_       = param.black_from();
  black_to_         = param.black_to();
  white_from_       = param.white_from();
  white_to_         = param.white_to();

  // setup interpolation matrix
  n_control_point_insertions_ = param.n_control_point_insertions();
  n_control_points_ = pow(2, n_control_point_insertions_) + 1;
  lut_size_         = param.lut_size();

  // Matrix for linear interpolation and linear extrapolation
  //
  const int n_extrapol    = lut_size_/2;
  const Dtype dx_lowres   = 1.0 / (n_control_points_ - 1);
  const Dtype dx_highres  = 1.0 / (lut_size_ - 1);
  const int lin_mat_ncols = n_control_points_;
  const int lin_mat_nrows = lut_size_ + 2 * n_extrapol;
  Dtype* lin_mat = new Dtype[lin_mat_nrows * lin_mat_ncols];

  CreateLinearInterpExtrapMatrix(n_control_points_, dx_lowres,
                                 lut_size_, dx_highres,
                                 n_extrapol, lin_mat);

  // Matrix for Gaussian smoothing
  //
    Dtype sigma = dx_lowres / 4;
  const int gauss_mat_ncols = lin_mat_nrows;
  const int gauss_mat_nrows = lut_size_;
  Dtype* gauss_mat = new Dtype[gauss_mat_nrows * gauss_mat_ncols];

  for (int row = 0; row < gauss_mat_nrows; ++row) {
    Dtype xout = row * dx_highres;
    Dtype rowsum = 0;
    for (int col = 0; col < gauss_mat_ncols; ++col) {
      Dtype xin = (col - n_extrapol) * dx_highres;
      Dtype v = exp( -0.5 * (xout - xin) * (xout - xin) / (sigma * sigma));
      gauss_mat[ row * gauss_mat_ncols + col] = v;
      rowsum += v;
    }
    for (int col = 0; col < gauss_mat_ncols; ++col) {
      gauss_mat[ row * gauss_mat_ncols + col] /= rowsum;
    }
  }

  // create interpolation matrix
  vector<int> interpol_mat_shape;
  interpol_mat_shape.push_back(lut_size_ * n_control_points_);
  interpol_mat_.Reshape(interpol_mat_shape);

  caffe_cpu_gemm<Dtype>( CblasNoTrans, CblasNoTrans,
                         lut_size_, n_control_points_, lin_mat_nrows,
                         Dtype(1), gauss_mat, lin_mat,
                         Dtype(0), interpol_mat_.mutable_cpu_data());
  delete[] lin_mat;
  delete[] gauss_mat;

  // control points of lookup table are sampled with a strategy
  // involving several user-set ranges and in some cases an incompatible
  // set of parameters is sampled.
  // Maximum 50% of failures are allowed.
  int n_fails = 0;
  for (int i=0; i < 100000; ++i) {
     try {
     std::vector<Dtype> lut = dense_lut( random_lut_controlpoints(
          black_from_, black_to_,
          white_from_, white_to_,
          slope_min_,  slope_max_,
         n_control_point_insertions_));
     } catch (const std::runtime_error& e) {
          n_fails++;
     }
  }

  LOG(INFO) << "Generating 100000 random lookup tables with given " <<
      "parameters for ValueAugmentationLayer.";
  LOG(INFO) << "Failure rate: " << (n_fails*100.)/100000 << "%";

  CHECK_LE(n_fails, 50000) << "Provided value augmentation parameters " <<
      "produced in more than 50% of cases incompatible control points." <<
      " Parameters of ValueAugmentationLayer need to be changed.";
}


template <typename Dtype>
std::vector<Dtype> ValueAugmentationLayer<Dtype>::random_lut_controlpoints(
    Dtype black_from, Dtype black_to,
    Dtype white_from, Dtype white_to,
    Dtype slope_min,  Dtype slope_max,
    int n_control_point_insertions) {
  // initial lut has only start and end point
  //
  std::vector<Dtype> lut(2);
  lut[0] = uniform_random_value( black_from, black_to);
  lut[1] = uniform_random_value( white_from, white_to);
  Dtype dx = 1;

  for (int iter = 0; iter < n_control_point_insertions; ++iter) {
    // insert intermediate points
    std::vector<Dtype> newlut;
    newlut.push_back(lut[0]);
    dx /= 2;
    for (int i = 0; i < lut.size() - 1; ++i) {
      Dtype left_constraint_from  = lut[i] + slope_min * dx;
      Dtype left_constraint_to    = lut[i] + slope_max * dx;
      Dtype right_constraint_from = lut[i+1] - slope_max * dx;
      Dtype right_constraint_to   = lut[i+1] - slope_min * dx;
      Dtype ymin = std::max( left_constraint_from, right_constraint_from);
      Dtype ymax = std::min( left_constraint_to, right_constraint_to);
      if (ymin > ymax) {
          throw std::runtime_error("Invalid arguments for subsequent call of "
                                   "uniform_random_value encountered.");
      }
      newlut.push_back( uniform_random_value( ymin, ymax));
      newlut.push_back( lut[i+1]);
    }
    lut = newlut;
  }

  return lut;
}

template <typename Dtype>
std::vector<Dtype> ValueAugmentationLayer<Dtype>::dense_lut(
    const std::vector<Dtype>& lut_control_points) {
  std::vector<Dtype> lut(lut_size_);
  caffe_cpu_gemv<Dtype>( CblasNoTrans,
                         lut_size_, n_control_points_,
                         Dtype(1), interpol_mat_.cpu_data(),
                         lut_control_points.data(), Dtype(0), lut.data());
  return lut;
}


template <typename Dtype>
void ValueAugmentationLayer<Dtype>::apply_lut(
    const std::vector<Dtype>& lut,
    const Dtype* in_data, Dtype* out_data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    float lut_i = in_data[i] * (lut.size()-1);
    if (lut_i <= 0) {
      out_data[i]  = lut[0];
    } else if (lut_i >= lut.size()-1) {
      out_data[i] = lut[lut.size()-1];
    } else {
      int i1  = floor( lut_i);
      float f = lut_i - i1;
      out_data[i] = (1-f) * lut[i1] + f * lut[i1+1];
    }
  }
}


template <typename Dtype>
void ValueAugmentationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int nsamples  = bottom[0]->shape(0);
  const int nchannels = bottom[0]->shape(1);
  const int count     = bottom[0]->count() / (nsamples * nchannels);

  for (int num = 0; num < nsamples; ++num) {
    for (int ch = 0; ch < nchannels; ++ch) {
      while (true) {
        try {
          std::vector<Dtype> lut = dense_lut( random_lut_controlpoints(
            black_from_, black_to_,
            white_from_, white_to_,
            slope_min_,  slope_max_,
            n_control_point_insertions_));
          apply_lut( lut, bottom_data + (num * nchannels + ch) * count,
                 top_data + (num * nchannels + ch) * count,
                 count);
          break;
        } catch (const std::runtime_error& e) {}
      }
    }
  }
}


template <typename Dtype>
void ValueAugmentationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_CLASS(ValueAugmentationLayer);
REGISTER_LAYER_CLASS(ValueAugmentation);

}  // namespace caffe

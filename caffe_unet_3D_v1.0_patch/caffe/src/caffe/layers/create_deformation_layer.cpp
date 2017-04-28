#include <vector>

#include "caffe/layers/create_deformation_layer.hpp"
#include "caffe/util/vector_helper.hpp"

namespace caffe {

template <typename Dtype>
void CreateDeformationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const CreateDeformationParameter& param =
      this->layer_param_.create_deformation_param();
  CHECK_EQ(1, top.size());
  CHECK( param.ncomponents() == 2
         || param.ncomponents() == 3)
      << "size of last axis (ncomponents) must be "
      "2 (for 2D deformation vectors) or 3 (for 3D deformation vectors)";
  if( param.nz() == 0 ) {
    CHECK( param.ncomponents() == 2)
        << "size of last axis (ncomponents) must be "
        "2 for 2D deformation fields";
  }

  CHECK_LE( param.random_offset_range_from_in_blob_shape()
            + param.random_offset_range_from_pdf()
            + (param.random_offset_range_from_ignore_label() > 0), 1) <<
      "Only one of 'random_offset_range_from_in_blob_shape', 'random_offset_range_from_pdf' or 'random_offset_range_from_ignore_label' can be selected!";
  //  CHECK( param.has_random_offset_range_from_ignore_label()) << "Not implemented yet";

  batch_size_ = 0;
  Reshape( bottom, top);
}

template <typename Dtype>
void CreateDeformationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
  const CreateDeformationParameter& param =
      this->layer_param_.create_deformation_param();

  // We only have to reshape the top blob and the internal arrays, if
  // the batch_size changed
  int new_batch_size = param.batch_size();
  if(bottom.size() == 1) {
    new_batch_size = bottom[0]->shape(0);
  }
  if( new_batch_size == batch_size_) return;
  batch_size_ = new_batch_size;

  // reshape top blob to requested shape
  vector<int> out_shape;
  out_shape.push_back(batch_size_);
  if( param.nz() > 0) out_shape.push_back(param.nz());
  out_shape.push_back(param.ny());
  out_shape.push_back(param.nx());
  out_shape.push_back(param.ncomponents());

  top[0]->Reshape( out_shape);

  n_spatial_axes_ = (param.nz() > 0)? 3 : 2;
  n_deform_comps_ = param.ncomponents();

  // check if elastic transformation is requested
  do_elastic_trafo_ = false;
  if( param.random_elastic_grid_spacing().v_size() > 0) {
    do_elastic_trafo_ = true;
    // create the cubic bspline interpolation kernels
    // for the smooth deformation field
    if( n_spatial_axes_ == 2) {
      CHECK_EQ(2, param.random_elastic_grid_spacing().v_size());
      grid_spacing_z_ = 0;
      grid_spacing_y_ = param.random_elastic_grid_spacing().v(0);
      grid_spacing_x_ = param.random_elastic_grid_spacing().v(1);
      bkernel_z_ = NULL;
    } else {
      CHECK_EQ(3, param.random_elastic_grid_spacing().v_size());
      grid_spacing_z_ = param.random_elastic_grid_spacing().v(0);
      grid_spacing_y_ = param.random_elastic_grid_spacing().v(1);
      grid_spacing_x_ = param.random_elastic_grid_spacing().v(2);
      bkernel_z_ = create_bspline_kernels( grid_spacing_z_);
    }
    bkernel_y_ = create_bspline_kernels( grid_spacing_y_);
    bkernel_x_ = create_bspline_kernels( grid_spacing_x_);

    // setup intermediate arrays for deformation field generation
    if( n_spatial_axes_ == 2) {
      int size_y    = top[0]->shape(1);
      int size_x    = top[0]->shape(2);
      int ncp_y     = size_y / grid_spacing_y_ + 4;
      int ncp_x     = size_x / grid_spacing_x_ + 4;
      rdispl_       = new Dtype[ncp_y * ncp_x];
      rdispl_shape_ = make_int_vect( ncp_y, ncp_x);
      tmp1_         = new Dtype[size_y * ncp_x];
      tmp1_shape_   = make_int_vect( size_y, ncp_x);
      tmp2_         = NULL;
    } else {
      int size_z    = top[0]->shape(1);
      int size_y    = top[0]->shape(2);
      int size_x    = top[0]->shape(3);
      int ncp_z     = size_z / grid_spacing_z_ + 4;
      int ncp_y     = size_y / grid_spacing_y_ + 4;
      int ncp_x     = size_x / grid_spacing_x_ + 4;
      rdispl_       = new Dtype[ncp_z * ncp_y * ncp_x];
      rdispl_shape_ = make_int_vect( ncp_z, ncp_y, ncp_x);
      tmp1_         = new Dtype[size_z * ncp_y * ncp_x];
      tmp1_shape_   = make_int_vect( size_z, ncp_y, ncp_x);
      tmp2_         = new Dtype[size_z * size_y * ncp_x];
      tmp2_shape_   = make_int_vect( size_z, size_y, ncp_x);
    }
  }

  // get the relative element_size in z-direction. Needed for rotation
  voxel_relsize_z_ = param.voxel_relsize_z();

  // get the range for the rotation angles
  //
  if( param.random_rotate_from().v_size() > 0) {
    if( n_deform_comps_ == 2) {
      // 2 component displacement
      CHECK_EQ(1, param.random_rotate_from().v_size()) << "for 2 components "
          "the rotation angle must be a single scalar";
      CHECK_EQ(1, param.random_rotate_to().v_size()) << "for 2 components "
          "the rotation angle must be a single scalar";
      rot_from_ = make_vec<float>(param.random_rotate_from().v(0), 0, 0);
      rot_to_   = make_vec<float>(param.random_rotate_to().v(0), 0, 0);
    } else {
      // 3 component displacement
      CHECK_EQ(3, param.random_rotate_from().v_size()) << "for 3 components "
          "the rotation angle must be a 3 component vector";
      CHECK_EQ(3, param.random_rotate_to().v_size()) << "for 3 components "
          "the rotation angle must be a 3 component vector";
      rot_from_ = make_vec<float>(param.random_rotate_from().v(0),
                                  param.random_rotate_from().v(1),
                                  param.random_rotate_from().v(2));
      rot_to_   = make_vec<float>(param.random_rotate_to().v(0),
                                  param.random_rotate_to().v(1),
                                  param.random_rotate_to().v(2));
    }
  } else {
    rot_from_ = make_vec<float>(0, 0, 0);
    rot_to_   = make_vec<float>(0, 0, 0);
  }

  // get the range for the offsets
  //
  if( param.random_offset_from().v_size() > 0) {
    if( n_deform_comps_ == 2) {
      // 2 component displacement
      CHECK_EQ(2, param.random_offset_from().v_size()) << "for 2 components "
          "the offset must be a 2 component vector";
      CHECK_EQ(2, param.random_offset_to().v_size()) << "for 2 components "
          "the offset must be a 2 component vector";
      offset_from_ = make_vec<float>(0,
                                     param.random_offset_from().v(0),
                                     param.random_offset_from().v(1));
      offset_to_   = make_vec<float>(0,
                                     param.random_offset_to().v(0),
                                     param.random_offset_to().v(1));
    } else {
      // 3 component displacement
      CHECK_EQ(3, param.random_offset_from().v_size()) << "for 3 components "
          "the offset must be a 3 component vector";
      CHECK_EQ(3, param.random_offset_to().v_size()) << "for 2 components "
          "the offset must be a 3 component vector";
      offset_from_ = make_vec<float>(param.random_offset_from().v(0),
                                     param.random_offset_from().v(1),
                                     param.random_offset_from().v(2));
      offset_to_   = make_vec<float>(param.random_offset_to().v(0),
                                     param.random_offset_to().v(1),
                                     param.random_offset_to().v(2));
    }
  } else {
    offset_from_ = make_vec<float>(0, 0, 0);
    offset_to_   = make_vec<float>(0, 0, 0);
  }


  // get the random mirror flags
  //
  if( param.random_mirror_flag().v_size() > 0) {
    if( n_deform_comps_ == 2) {
      // 2 component displacement
      CHECK_EQ(2, param.random_mirror_flag().v_size()) << "for 2 components "
          "the mirror flag must be a 2 component vector";
      mirror_flag_ = make_vec<int>(0,
                                   param.random_mirror_flag().v(0),
                                   param.random_mirror_flag().v(1));
    } else {
      // 3 component displacement
      CHECK_EQ(3,  param.random_mirror_flag().v_size()) << "for 3 components "
          "the mirror flag must be a 3 component vector";
      mirror_flag_ = make_vec<int>(param.random_mirror_flag().v(0),
                                   param.random_mirror_flag().v(1),
                                   param.random_mirror_flag().v(2));
    }
  } else {
    mirror_flag_ = make_vec<int>(0, 0, 0);
  }


}

template <typename Dtype>
void CreateDeformationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const CreateDeformationParameter& param =
      this->layer_param_.create_deformation_param();
  //  std::cout << "CreateDeformationLayer<Dtype>::Forward_cpu\n";

  if( do_elastic_trafo_) {
    if (n_spatial_axes_ == 2) {
      const int top_ny    = top[0]->shape(1);
      const int top_nx    = top[0]->shape(2);
      const int top_ncomp = top[0]->shape(3);
      for (int n = 0; n < top[0]->shape(0); ++n) {
        for (int c = 0; c < top_ncomp; ++c) {
          //          std::cout << "n=" << n << ", c=" << c <<std::endl;
          // create random displacements on grid
          caffe_rng_gaussian(
              rdispl_shape_[0] * rdispl_shape_[1],
              Dtype(0), Dtype(param.random_elastic_deform_magnitude().v(c)),
              rdispl_);
          // scale up in y direction with bspline kernel
          cubic_bspline_interpolation(
              rdispl_, rdispl_shape_[1], rdispl_shape_[0],
              1, rdispl_shape_[1],
              tmp1_, tmp1_shape_[0],
              1, tmp1_shape_[1],
              bkernel_y_, grid_spacing_y_);
          // scale up in x direction with bspline kernel
          cubic_bspline_interpolation(
              tmp1_, tmp1_shape_[0], tmp1_shape_[1],
              tmp1_shape_[1], 1,
              top[0]->mutable_cpu_data() + top[0]->offset(make_int_vect(n,0,0,c)),
              top_nx,
              top_nx * top_ncomp, top_ncomp,
              bkernel_x_, grid_spacing_x_);
        }
        // add the unit transform
        Dtype* p = top[0]->mutable_cpu_data()
            + top[0]->offset(make_int_vect(n,0,0,0));
        for (int y = 0; y < top_ny; ++y) {
          for (int x = 0; x < top_nx; ++x) {
            p[0] += y;
            p[1] += x;
            p += 2;
          }
        }

      }
    } else {
      const int top_nz    = top[0]->shape(1);
      const int top_ny    = top[0]->shape(2);
      const int top_nx    = top[0]->shape(3);
      const int top_ncomp = top[0]->shape(4);
      for (int n = 0; n < top[0]->shape(0); ++n) {
        for( int c = 0; c < top_ncomp; ++c) {
          //         std::cout << "n=" << n << ", c=" << c <<std::endl;
          // create random displacements on grid
          caffe_rng_gaussian(
              rdispl_shape_[0] * rdispl_shape_[1] * rdispl_shape_[2],
              Dtype(0), Dtype(param.random_elastic_deform_magnitude().v(c)),
              rdispl_);
          //
          // scale up in z direction with bspline kernel
          const Dtype* in      = rdispl_;
          int in_n_lines       = rdispl_shape_[1] * rdispl_shape_[2];
          int in_n_elem        = rdispl_shape_[0];
          int in_stride_lines  = 1;
          int in_stride_elem   = rdispl_shape_[1] * rdispl_shape_[2];
          Dtype* out           = tmp1_;
          int out_n_elem       = tmp1_shape_[0];
          int out_stride_lines = 1;
          int out_stride_elem  = tmp1_shape_[1] * tmp1_shape_[2];
          cubic_bspline_interpolation(
              in, in_n_lines, in_n_elem, in_stride_lines, in_stride_elem,
              out, out_n_elem, out_stride_lines, out_stride_elem,
              bkernel_z_, grid_spacing_z_);
          //
          // scale up in y direction with bspline kernel
          for (int z = 0; z < top_nz; ++z) {
            in               = tmp1_ + z * (tmp1_shape_[1] * tmp1_shape_[2]);
            in_n_lines       = tmp1_shape_[2];
            in_n_elem        = tmp1_shape_[1];
            in_stride_lines  = 1;
            in_stride_elem   = tmp1_shape_[2];
            out              = tmp2_ + z * (tmp2_shape_[1] * tmp2_shape_[2]);
            out_n_elem       = tmp2_shape_[1];
            out_stride_lines = 1;
            out_stride_elem  = tmp2_shape_[2];
            cubic_bspline_interpolation(
                in, in_n_lines, in_n_elem, in_stride_lines, in_stride_elem,
                out, out_n_elem, out_stride_lines, out_stride_elem,
                bkernel_y_, grid_spacing_y_);
          }
          // scale up in x direction with bspline kernel
          in               = tmp2_;
          in_n_lines       = tmp2_shape_[0] * tmp2_shape_[1];
          in_n_elem        = tmp2_shape_[2];
          in_stride_lines  = tmp2_shape_[2];
          in_stride_elem   = 1;
          out              = top[0]->mutable_cpu_data()
              + top[0]->offset(make_int_vect(n,0,0,0,c));
          out_n_elem       = top_nx;
          out_stride_lines = top_nx * top_ncomp;
          out_stride_elem  = top_ncomp;
          //        std::cout << "scale in x-direction\n"
          //            "offset: " << top[0]->offset(make_int_vect(n,0,0,0,c)) << "\n";
          cubic_bspline_interpolation(
              in, in_n_lines, in_n_elem, in_stride_lines, in_stride_elem,
              out, out_n_elem, out_stride_lines, out_stride_elem,
              bkernel_x_, grid_spacing_x_);
        }
        // add the unit transform
        Dtype* p = top[0]->mutable_cpu_data()
            + top[0]->offset(make_int_vect(n,0,0,0,0));
        for (int z = 0; z < top_nz; ++z) {
          for (int y = 0; y < top_ny; ++y) {
            for (int x = 0; x < top_nx; ++x) {
              p[0] += z;
              p[1] += y;
              p[2] += x;
              p += 3;
            }
          }
        }
      }
    }
  } else {
    // no elastic trafo requested. Fill deformation field with
    // identity transform
    if( n_spatial_axes_ == 2) {
      // 2D, 2 component deformation field
      Dtype* p = top[0]->mutable_cpu_data();
      for (int n = 0; n < top[0]->shape(0); ++n) {
        for (int y = 0; y < top[0]->shape(1); ++y) {
          for (int x = 0; x < top[0]->shape(2); ++x) {
            p[0] = y;
            p[1] = x;
            p += 2;
          }
        }
      }
    } else {
      if (n_deform_comps_ == 2) {
        // 3D, 2 component deformation field
        Dtype* p = top[0]->mutable_cpu_data();
        for (int n = 0; n < top[0]->shape(0); ++n) {
          for (int z = 0; z < top[0]->shape(1); ++z) {
            for (int y = 0; y < top[0]->shape(2); ++y) {
              for (int x = 0; x < top[0]->shape(3); ++x) {
                p[0] = y;
                p[1] = x;
                p += 2;
              }
            }
          }
        }
      } else {
        // 3D, 3 component deformation field
        Dtype* p = top[0]->mutable_cpu_data();
        for (int n = 0; n < top[0]->shape(0); ++n) {
          for (int z = 0; z < top[0]->shape(1); ++z) {
            for (int y = 0; y < top[0]->shape(2); ++y) {
              for (int x = 0; x < top[0]->shape(3); ++x) {
                p[0] = z;
                p[1] = y;
                p[2] = x;
                p += 3;
              }
            }
          }
        }
      }
    }
  }

  // find out the bottom shape
  int bottom_nz = 1;
  int bottom_ny = 1;
  int bottom_nx = 1;

  if( bottom.size() == 0) {
    // If bottom shape is unkown, we assume the same shape as for top
    if (n_spatial_axes_ == 2) {
      bottom_ny = top[0]->shape(1);
      bottom_nx = top[0]->shape(2);
    } else {
      bottom_nz = top[0]->shape(1);
      bottom_ny = top[0]->shape(2);
      bottom_nx = top[0]->shape(3);
    }
  } else {
    // if a bottom blob is there. Take its shape
    if (n_spatial_axes_ == 2) {
      bottom_ny = bottom[0]->shape(2);
      bottom_nx = bottom[0]->shape(3);
    } else {
      bottom_nz = bottom[0]->shape(2);
      bottom_ny = bottom[0]->shape(3);
      bottom_nx = bottom[0]->shape(4);
    }
  }

  // top shape for convenience
  int top_nz = 1;
  int top_ny = 1;
  int top_nx = 1;
  if (n_spatial_axes_ == 2) {
    top_ny = top[0]->shape(1);
    top_nx = top[0]->shape(2);
  } else {
    top_nz = top[0]->shape(1);
    top_ny = top[0]->shape(2);
    top_nx = top[0]->shape(3);
  }


  // for all images in the batch
  for (int batchIdx = 0; batchIdx < top[0]->shape(0); ++batchIdx) {
    // create the 4x4 transformation matrix (homgogenous coordinates)
    // first draw the random rotation angle, offset and mirrorfactor
    vector<float> angle(3);
    vector<float> offset(3);
    vector<float> mirrorfactor(3);
    for (int i = 0; i < 3; ++i) {
      caffe_rng_uniform( 1, rot_from_[i], rot_to_[i], &angle[i]);
      caffe_rng_uniform( 1, offset_from_[i], offset_to_[i], &offset[i]);
      if( mirror_flag_[i] == 1) {
        int b;
        caffe_rng_bernoulli( 1, 0.5, &b);
        mirrorfactor[i] = (b==0)? -1 : 1;
      } else {
        mirrorfactor[i] = 1;
      }
    }

    // if a offset sampling from input shape is requested, do it
    if( param.random_offset_range_from_in_blob_shape()) {
      int z = caffe_rng_rand() % bottom_nz;
      int y = caffe_rng_rand() % bottom_ny;
      int x = caffe_rng_rand() % bottom_nx;

      offset[0] += z - float(bottom_nz - 1) / 2;
      offset[1] += y - float(bottom_ny - 1) / 2;
      offset[2] += x - float(bottom_nx - 1) / 2;
      std::cout << "random_offset_range_from_in_blob_shape offset: " << toString(offset) << std::endl;
    }

    // if a probability map for offsets is given, use it to sample an
    // offset
    if( param.random_offset_range_from_pdf()) {
      size_t pdf_size = bottom[0]->count() / bottom[0]->shape(0);
      const Dtype* pdf = bottom[0]->cpu_data() + batchIdx * pdf_size;

      std::vector<Dtype> cdf(pdf_size);
      caffe_cpu_cumsum( pdf_size, pdf, cdf.data());
      int x, y, z;
      caffe_rand_pos_arbitrary_cdf( cdf.data(), bottom_nz, bottom_ny, bottom_nx,
                                    &z, &y, &x);
      offset[0] += z - float(bottom_nz - 1) / 2;
      offset[1] += y - float(bottom_ny - 1) / 2;
      offset[2] += x - float(bottom_nx - 1) / 2;
      //  std::cout << "offset " << toString(offset) << std::endl;
   }

    // ** orig 3D unet **
    // if a label map for offsets is given, use non-ignore labels to
    // sample an offset
    if( param.random_offset_range_from_ignore_label() > 0) {
      size_t labels_size = bottom[0]->count() / bottom[0]->shape(0);
      const Dtype* labels = bottom[0]->cpu_data() + batchIdx * labels_size;
      int ignore_label = param.random_offset_range_from_ignore_label();

      // compute cummulative sum of non-ignore-label-pdf
      std::vector<Dtype> cdf(labels_size);
      Dtype cumsum = 0;
      for (size_t i = 0; i < labels_size; ++i) {
        cumsum += (int(labels[i]) != ignore_label);
        cdf[i] = cumsum;
      }
      //   std::cout << "cumsum = " << cumsum << std::endl;

      int x, y, z;
      caffe_rand_pos_arbitrary_cdf( cdf.data(), bottom_nz, bottom_ny, bottom_nx,
                                    &z, &y, &x);
      offset[0] += z - float(bottom_nz - 1) / 2;
      offset[1] += y - float(bottom_ny - 1) / 2;
      offset[2] += x - float(bottom_nx - 1) / 2;
        std::cout << "offset from ignore_label " << ignore_label << ": " << toString(offset) << std::endl;
   }

    // **Holger: orig version seems to work fine and faster..***
    // if a label map for offsets is given, use non-ignore labels to
    // sample an offset
    /*if( param.random_offset_range_from_ignore_label() > 0) {
      std::cout << "bottom[0]->count() " << bottom[0]->count() << std::endl;
      std::cout << "bottom[0]->shape() " << bottom[0]->shape() << std::endl;
      size_t labels_size = bottom[0]->count() / bottom[0]->shape(0);
      const Dtype* labels = bottom[0]->cpu_data() + batchIdx * labels_size;
      int ignore_label = param.random_offset_range_from_ignore_label();
      std::vector<unsigned int> idx3D(3);
      std::vector< std::vector<unsigned int> > indices3D;
      indices3D.clear();
      // iterate through label image
      unsigned long i = 0;
      int max_labels = 0;
      printf("bottom_nx/y/z %d, %d, %d (batchIdx %d)\n",bottom_nx,bottom_ny,bottom_nz,batchIdx);
      for (unsigned int x = 0; x < bottom_nx; ++x) 
      {
	for (unsigned int y = 0; y < bottom_ny; ++y) 
        {
	   for (unsigned int z = 0; z < bottom_nz; ++z) 
           { 
               //std::cout << labels[i] << " vs " << ignore_label << ", ";  
	       //printf("%d. %d,%d,%d, label %d, ignore %d\n",i,x,y,z,labels[i],int(ignore_label));
	       if (labels[i] > max_labels)
	       {
		 max_labels = labels[i];
               }		
	       if (int(labels[i]) != int(ignore_label))
	       {  
		   idx3D[0] = x;
		   idx3D[1] = y;
		   idx3D[2] = z;
		   indices3D.push_back(idx3D);	
	       }
	       else
               { 
                 std::cout << "NOT: " << int(labels[i]) << " vs " << int(ignore_label) << std::endl;  
               } 
	       i++;
	   }
	}
      }
      printf("  %d of %d are non-ignore voxels (%g perc.), max label: %d.\n", indices3D.size(), labels_size, 100*float(indices3D.size())/labels_size, max_labels);
       
      // get random index
      Dtype r;
      caffe_rng_uniform( 1, Dtype(0), Dtype(1), &r);
      unsigned long ridx = r*labels_size;
      std::cout << "  random (" << r << ") index: " << ridx << std::endl;
      idx3D = indices3D[ridx];
      int x = idx3D[0];
      int y = idx3D[1];
      int z = idx3D[2];

      offset[0] += z - float(bottom_nz - 1) / 2;
      offset[1] += y - float(bottom_ny - 1) / 2;
      offset[2] += x - float(bottom_nx - 1) / 2;
      std::cout << "offset from ignore_label " << ignore_label << ": " << toString(offset) << std::endl;
   }*/

    // start with the unit matrix
    vector<float> M = make_vec<float>(
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1);

    // shift target center to origin
    M = m_shift3D<float>( -float(top_nz) / 2,
                          -float(top_ny) / 2,
                          -float(top_nx) / 2, M);

    // scale to cubic voxels
    M = m_scale3D<float>( voxel_relsize_z_, 1, 1, M);

    // do the rotation
    M = m_rotate3D<float>( angle[0], angle[1], angle[2], M);

    // scale back to original voxel size
    M = m_scale3D<float>( 1.0/voxel_relsize_z_, 1, 1, M);

    // shift origin to src center
    M = m_shift3D<float>( float(bottom_nz) / 2,
                          float(bottom_ny) / 2,
                          float(bottom_nx) / 2, M);

    // apply the offset
    std::cout << " apply offset: " << toString(offset) << std::endl;
    M = m_shift3D<float>( offset[0], offset[1], offset[2], M);

    //    std::cout << "resulting matrix:\n" << Array2DtoString( M.data(), 4, 4);


    // transform all deformation vectors with the resulting matrix
    size_t out_data_size = top[0]->count() / top[0]->shape(0);
    Dtype* out_data = top[0]->mutable_cpu_data() + batchIdx * out_data_size;
    if (n_deform_comps_ == 2) {
      float a11 = M[5]; float a12 = M[6];  float b1 = M[7];
      float a21 = M[9]; float a22 = M[10]; float b2 = M[11];
      Dtype* p = out_data;
      size_t n_vectors = out_data_size / 2;
      for (int i = 0; i < n_vectors; ++i) {
        float v1 = a11*p[0] + a12*p[1] + b1;
        float v2 = a21*p[0] + a22*p[1] + b2;
        p[0] = v1;
        p[1] = v2;
        p += 2;
      }
    } else {
      float a11 = M[0]; float a12 = M[1]; float a13 = M[2]; float b1 = M[3];
      float a21 = M[4]; float a22 = M[5]; float a23 = M[6]; float b2 = M[7];
      float a31 = M[8]; float a32 = M[9]; float a33 = M[10]; float b3 = M[11];
      Dtype* p = out_data;
      size_t n_vectors = out_data_size / 3;
      for (int i = 0; i < n_vectors; ++i) {
        float v1 = a11*p[0] + a12*p[1] + a13*p[2] + b1;
        float v2 = a21*p[0] + a22*p[1] + a23*p[2] + b2;
        float v3 = a31*p[0] + a32*p[1] + a33*p[2] + b3;
        p[0] = v1;
        p[1] = v2;
        p[2] = v3;
        p += 3;
      }
    }
  }

}

template <typename Dtype>
Dtype* CreateDeformationLayer<Dtype>::create_bspline_kernels(int nb) {
  Dtype* b0123 = new Dtype[4*nb];
  Dtype* b0 = b0123;
  Dtype* b1 = b0123 + nb;
  Dtype* b2 = b0123 + 2 * nb;
  Dtype* b3 = b0123 + 3 * nb;

  for (int i = 0; i < nb; ++i) {
    Dtype x = Dtype(i) / nb;
    b0[i] = 1./6 * ( - x*x*x +  3*x*x - 3*x + 1);
    b1[i] = 1./6 * ( 3*x*x*x + -6*x*x       + 4);
    b2[i] = 1./6 * (-3*x*x*x +  3*x*x + 3*x + 1);
    b3[i] = 1./6 * x*x*x;
  }
  return b0123;
}


template <typename Dtype>
void CreateDeformationLayer<Dtype>::cubic_bspline_interpolation(
    const Dtype* in, int in_n_lines, int in_n_elem,
    int in_stride_lines, int in_stride_elem,
    Dtype* out, int out_n_elem, int out_stride_lines, int out_stride_elem,
    const Dtype* b0123, int nb) {
  const Dtype* b0 = b0123;
  const Dtype* b1 = b0123 + nb;
  const Dtype* b2 = b0123 + 2 * nb;
  const Dtype* b3 = b0123 + 3 * nb;
//  std::cout << "cubic_bspline_interpolation() \n"
//      << "in_n_lines,      " << in_n_lines << std::endl
//      << "in_n_elem,       " << in_n_elem << std::endl
//      << "in_stride_lines, " << in_stride_lines << std::endl
//      << "in_stride_elem,  " << in_stride_elem << std::endl
//      << "out_n_elem,      " << out_n_elem << std::endl
//      << "out_stride_lines," << out_stride_lines << std::endl
//      << "out_stride_elem, " << out_stride_elem << std::endl
//      << "nb               " << nb               << std::endl;
  for (int line_i = 0; line_i < in_n_lines; ++line_i) {
    int out_elem_i = 0;
    for (int elem_i = 0; elem_i < in_n_elem - 3; ++elem_i) {
      int in_offs = line_i * in_stride_lines + elem_i * in_stride_elem;
      Dtype w0 = in[in_offs];
      Dtype w1 = in[in_offs + in_stride_elem];
      Dtype w2 = in[in_offs + 2 * in_stride_elem];
      Dtype w3 = in[in_offs + 3 * in_stride_elem];
      Dtype* out_p = out + line_i * out_stride_lines
          + elem_i * (out_stride_elem * nb);
      for( int i = 0; i < nb && out_elem_i < out_n_elem; ++i, ++out_elem_i) {
        *out_p =  w0 * b0[i] + w1 * b1[i] + w2 * b2[i] + w3 * b3[i];
        out_p += out_stride_elem;
      }
    }
  }
}




INSTANTIATE_CLASS(CreateDeformationLayer);

REGISTER_LAYER_CLASS(CreateDeformation);

}  // namespace caffe

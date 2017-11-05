//
// Created by slam on 17-10-30.
//

#ifndef PROJECT_HDF5_CT_H
#define PROJECT_HDF5_CT_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/algorithm/string.hpp>

#include "hdf5.h"
#include "hdf5_hl.h"

class Mat2H5 {
public:
    Mat2H5(float mean_val = 127, float var = 1/128.0f)
            :mean_value_(mean_val), shrink_ratio_(var){};
    struct h5dataset {
        std::string datasetname;
        std::string datatype;
        int rank;
        std::vector<hsize_t > dimension;
    };
    void create_hdf5(std::string);
    void open_hdf5(std::string);
    void close_hdf5();
    void write_mat2hdf5(h5dataset, const std::vector<cv::Mat>&);
    void write_array2hdf5(h5dataset, const float*);
private:
    void transfer2array(float*, const std::vector<cv::Mat>&);
private:
    hid_t h5_fid_;
    float mean_value_;
    float shrink_ratio_;
};

#endif //PROJECT_HDF5_CT_H

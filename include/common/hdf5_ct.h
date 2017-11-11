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
#include "H5Cpp.h"

class Mat2H5 {
public:
    Mat2H5(float mean_val = 0, float var = 1.0f, int chunk_size = 100)
            :mean_value_(mean_val), shrink_ratio_(var),
             chunk_size_(chunk_size), data_offset_(0), label_offset_(0),
             data_type_(H5::PredType::NATIVE_FLOAT), label_type_(H5::PredType::NATIVE_FLOAT){};
    enum DataName {
        DATA = 3,
        LABEL = 4
    };
    void create_hdf5(std::string);
    void create_dataset(DataName, std::vector<int>, std::string);
    void open_hdf5(std::string);
    void close_hdf5();
    void write_data2hdf5(const std::vector<cv::Mat>&);
    void write_label2hdf5(const float*, int);
private:
    void transfer2array(float*, const std::vector<cv::Mat>&);
    H5::PredType get_type(std::string);
private:
    H5::H5File file_;
    float mean_value_;
    float shrink_ratio_;
    hsize_t data_offset_;
    hsize_t label_offset_;
    hsize_t chunk_size_;
    boost::shared_ptr<H5::DataSet>  dataset_data_;
    std::vector<hsize_t > data_dims_;
    boost::shared_ptr<H5::DataSet>  dataset_label_;
    std::vector<hsize_t> label_dims_;
    H5::PredType data_type_;
    H5::PredType label_type_;
private:
};

#endif //PROJECT_HDF5_CT_H

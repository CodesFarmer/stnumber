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
    struct h5dataset {
        std::string datasetname;
        std::string datatype;
        int rank;
        std::vector<hsize_t > dimension;
//        H5::PredType get_data_type() {
//            if(boost::iequals(datatype, "float"))
//                return H5::PredType::NATIVE_FLOAT;
//            if(boost::iequals(datatype, "int"))
//                return H5::PredType::NATIVE_INT;
//            if(boost::iequals(datatype, "uint8"))
//                return H5::PredType::NATIVE_UINT8;
//            if(boost::iequals(datatype, "double"))
//                return H5::PredType::NATIVE_DOUBLE;
//            if(boost::iequals(datatype, "uint16"))
//                return H5::PredType::NATIVE_UINT16;
//            if(boost::iequals(datatype, "uint32"))
//                return H5::PredType::NATIVE_UINT32;
//        }
    };
    void create_hdf5(std::string);
    void open_hdf5(std::string);
    void close_hdf5();
    void write2hdf5(h5dataset, const std::vector<cv::Mat>&);
private:
    void transfer2array(float*, std::vector<cv::Mat>&);
private:
    hid_t h5_fid_;
};

#endif //PROJECT_HDF5_CT_H

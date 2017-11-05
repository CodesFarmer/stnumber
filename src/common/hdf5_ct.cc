#include "hdf5_ct.h"

void Mat2H5::create_hdf5(std::string h5_filename) {
    if(access(h5_filename.c_str(), F_OK) != -1) {
        std::cout<<"Are you surely want to replace "<< h5_filename<<" with a new one?(yes/no)"<<std::endl;
        std::string confirm;
        std::cin>>confirm;
        if(!boost::iequals(confirm, "yes") && !boost::iequals(confirm, "Y")) {
            std::cout<<"Abort to generate a new HDF5 file..."<<std::endl;
            return;
        }
        std::cout<<"Replace the existing HDF5 file with new one..."<<std::endl;
    }
    //create the hdf5 file
    h5_fid_ = H5Fcreate(h5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if(h5_fid_ == 0) {
        std::cout<<"Failed to Generate a HDF5 file..."<<std::endl;
    }
    herr_t status = H5Fclose(h5_fid_);
    if(status != 0) {
        std::cout<<"Failed to close HDF5 file..."<<std::endl;
    }
}

void Mat2H5::open_hdf5(std::string h5_filename) {
    if(access(h5_filename.c_str(), F_OK) == -1) {
        std::cout<<"Make sure that you are open an existing HDF5 file..."<<std::endl;
    }
    h5_fid_ = H5Fopen(h5_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
}

void Mat2H5::close_hdf5() {
    herr_t status = H5Fclose(h5_fid_);
    if(status != 0) {
        std::cout<<"Failed to close HDF5 file..."<<std::endl;
    }
}

void Mat2H5::write_mat2hdf5(h5dataset data_sets, const std::vector<cv::Mat> &sources) {
    std::string dataset_name;
    dataset_name = data_sets.datasetname;
    int num_axes = data_sets.rank;
    hsize_t *dims = new hsize_t[num_axes];
    unsigned int data_num = 1;
    for(int iter = 0; iter < num_axes; iter++) {
        dims[iter] = data_sets.dimension[iter];
        data_num = data_num * dims[iter];
    }
    float *data = new float[data_num];
    transfer2array(data, sources);
    herr_t status = H5LTmake_dataset_float(
            h5_fid_, dataset_name.c_str(), num_axes, dims, data);
}

void Mat2H5::write_array2hdf5(h5dataset data_sets, const float* data_array) {
    std::string dataset_name;
    dataset_name = data_sets.datasetname;
    int num_axes = data_sets.rank;
    hsize_t *dims = new hsize_t[num_axes];
    unsigned int data_num = 1;
    for(int iter = 0; iter < num_axes; iter++) {
        dims[iter] = data_sets.dimension[iter];
        data_num = data_num * dims[iter];
//        std::printf("dimension : %d ", data_sets.dimension[iter]);
    }
    herr_t status = H5LTmake_dataset_float(
            h5_fid_, dataset_name.c_str(), num_axes, dims, data_array);
}

void Mat2H5::transfer2array(float *dst_array, const std::vector<cv::Mat> &sources) {
    int num_patch = sources.size();
    int num_channels = sources[0].channels();
    int height = sources[0].rows;
    int width = sources[0].cols;
//    std::printf("h:%d, w:%d, c:%d, n:%d\n", height, width, num_channels, num_patch);
    for(int iter_n = 0; iter_n<num_patch; iter_n++) {
//        std::cout<<"The size of image is "<<sources[iter_n].size()<<std::endl;
        cv::Mat img_tmp;
        sources[iter_n].convertTo(img_tmp, CV_32FC1);
//        std::cout<<"The size of image is "<<img_tmp.size()<<std::endl;
        for(int iter_c = 0; iter_c<num_channels ; iter_c++) {
            for(int iter_h = 0; iter_h<height; iter_h++) {
                for(int iter_w = 0; iter_w<width; iter_w++) {
                    int index = ( (iter_n*num_channels + iter_c)*height + iter_h)*width + iter_w;
                    float elem = (img_tmp.at<float>(iter_h, iter_w, iter_c) - mean_value_)*shrink_ratio_;
                    *(dst_array + index) = elem;
                }
            }
        }
    }
}
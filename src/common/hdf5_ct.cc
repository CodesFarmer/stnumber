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
//        boost::shared_ptr<H5::H5File> h5file_;
//        h5file_ = boost::make_shared<H5::H5File>(h5_filename, H5F_ACC_TRUNC);
//        //set the data type in hdf5
//        int fillvalue = 0;
//        H5::DSetCreatPropList plist;
//        plist.setFillValue(datasets.get_data_type(), &fillvalue);
//        //set the dimension of space
//        hsize_t *fdim = new hsize_t[datasets.rank];
//        for(int iter = 0; iter < datasets.rank; iter++) {
//            fdim[iter] = datasets.dimension[iter];
//        }
//        int space_rank = 4;
//        H5::DataSpace fspace(datasets.rank, fdim);
//        H5::DataSet * dataset_ptr;
//        dataset_ptr = new H5::DataSet(h5file_->createDataSet(datasets.datasetname,
//                                                         datasets.get_data_type(),
//                                                         fspace, plist));
//        delete dataset_ptr;
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

void Mat2H5::write2hdf5(h5dataset data_sets, const std::vector<cv::Mat> &sources) {
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
    herr_t status = H5LTmake_dataset_float(
            h5_fid_, dataset_name.c_str(), num_axes, dims, data);
}

void Mat2H5::transfer2array(float *dst_array, std::vector<cv::Mat> &sources) {
}
//    void write2hdf5(std::string h5_filename, const std::vector<cv::Mat> &sources) {
////        H5::H5File * h5file_ = new H5::H5File(h5_filename.c_str(), H5F_ACC_RDWR);
//        boost::shared_ptr<H5::H5File> h5file_;
//        h5file_ = boost::make_shared<H5::H5File>(h5_filename, H5F_ACC_RDWR);
//        //set the data type in hdf5
//        int fillvalue = 0;
//        H5::DSetCreatPropList plist;
//        plist.setFillValue(H5::PredType::NATIVE_FLOAT, &fillvalue);
//        //set the dimension of space
//        hsize_t fdim[] = {12, 12, 1, -1};
//        int space_rank = 4;
//        H5::DataSpace fspace(space_rank, fdim);
//        H5::DataSet * dataset;
//        dataset = new H5::DataSet(h5file_->createDataSet(dataset));
//    }
//}
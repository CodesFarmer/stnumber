#include "hdf5_ct.h"

void Mat2H5::create_hdf5(std::string h5_filename) {
    if (access(h5_filename.c_str(), F_OK) != -1) {
        std::cout << "Are you surely want to replace " << h5_filename << " with a new one?(yes/no)" << std::endl;
        std::string confirm;
        std::cin >> confirm;
        if (!boost::iequals(confirm, "yes") && !boost::iequals(confirm, "Y")) {
            std::cout << "Abort to generate a new HDF5 file..." << std::endl;
            return;
        }
        std::cout << "Replace the existing HDF5 file with new one..." << std::endl;
    }

    H5::H5File file(h5_filename, H5F_ACC_TRUNC);
    file_ = file;
}

void Mat2H5::create_dataset(DataName data_name, std::vector<int> dimension, std::string data_type) {
    std::string dataset_name;
    if(data_name == DATA) dataset_name = "data";
    else if(data_name == LABEL) dataset_name = "label";
    int num_axes = dimension.size();
    hsize_t *dims = new hsize_t[num_axes];
    hsize_t *max_dims = new hsize_t[num_axes];
    hsize_t *chunk_dims = new hsize_t[num_axes];
    unsigned int data_num = 1;
    for (int iter = 0; iter < num_axes; iter++) {
        dims[iter] = dimension[iter];
        max_dims[iter] = dimension[iter];
        chunk_dims[iter] = dimension[iter];
        data_num = data_num * dims[iter];
    }
    max_dims[0] = H5S_UNLIMITED;
    dims[0] = 0;
    //set the dataspace
    H5::DataSpace *dataspace = new H5::DataSpace(num_axes, dims, max_dims);
    //Set the property lists
    H5::DSetCreatPropList prop_list;
    chunk_dims[0] = chunk_size_;
    prop_list.setChunk(num_axes, chunk_dims);
    //create dataset with the data name
    boost::shared_ptr<H5::DataSet> dataset_h5;
    dataset_h5 = boost::make_shared<H5::DataSet>(
            file_.createDataSet(dataset_name,
                               get_type(data_type),
                               *dataspace, prop_list));
    if(data_name == DATA) {
        dataset_data_ = dataset_h5;
        for (int iter = 0; iter < num_axes; iter++) {
            data_dims_.push_back(dims[iter]);
        }
        data_type_ = get_type(data_type);
        //Adjust the size of mean value and variance
        if(mean_value_.size() == 1 && dims[1] != 1) {
            float mean_value_elem = mean_value_[0];
            mean_value_.resize(dims[1]);
            for(int iter = 0; iter < int(mean_value_.size()); iter++) {
                mean_value_[iter] = mean_value_elem;
            }
        }
        if(shrink_ratio_.size() == 1 && dims[1] != 1) {
            float shrink_ratio__elem = shrink_ratio_[0];
            shrink_ratio_.resize(dims[1]);
            for(int iter = 0; iter < int(shrink_ratio_.size()); iter++) {
                shrink_ratio_[iter] = shrink_ratio__elem;
            }
        }
    }
    else if(data_name == LABEL) {
        dataset_label_ = dataset_h5;
        for (int iter = 0; iter < num_axes; iter++) {
            label_dims_.push_back(dims[iter]);
        }
        label_type_ = get_type(data_type);
    }
    prop_list.close();

    delete [] max_dims;
    delete [] chunk_dims;
    delete dataspace;
}

void Mat2H5::open_hdf5(std::string h5_filename) {
    if(access(h5_filename.c_str(), F_OK) == -1) {
        std::cout<<"Make sure that you are open an existing HDF5 file..."<<std::endl;
    }
    file_ = H5Fopen(h5_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
}

void Mat2H5::close_hdf5() {
    file_.close();
}

void Mat2H5::write_data2hdf5(const std::vector<cv::Mat> &sources) {
    //extend data set first
    data_dims_[0] = data_dims_[0] + hsize_t(sources.size());
    dataset_data_->extend(&data_dims_[0]);
    //extend space
    H5::DataSpace *filespace = new H5::DataSpace(dataset_data_->getSpace());
    int num_axes = data_dims_.size();
    std::vector<hsize_t> dims_ext(num_axes, 0);
    for(int iter = 0;iter<num_axes;iter++ ) {
        dims_ext[iter] = data_dims_[iter];
    }
    dims_ext[0] = hsize_t(sources.size());
    std::vector<hsize_t> offset(num_axes, 0);
    offset[0] = data_offset_;
    filespace->selectHyperslab(H5S_SELECT_SET, &dims_ext[0], &offset[0]);
    //create new data space
    H5::DataSpace *memspace = new H5::DataSpace(num_axes, &dims_ext[0], NULL);
    //write the data into dataset
    int data_ext_num = 1;
    for(int iter = 0;iter<num_axes;iter++ ) {
        data_ext_num *= dims_ext[iter];
    }
    float *data_ext = new float[data_ext_num];
//    std::cout<<"The size of data extended is "<<data_ext_num<<std::endl;
    transfer2array(data_ext, sources);
    dataset_data_->write(data_ext, data_type_, *memspace, *filespace);
    data_offset_ = data_offset_ + hsize_t(sources.size());
    //free memories
    delete [] data_ext;
    delete memspace;
    delete filespace;
}

void Mat2H5::write_label2hdf5(const std::vector<std::vector<float> > &data_array) {
    //Check the dimension of label
    int label_dismension = 1;
    for(int iter = 1; iter < label_dims_.size(); iter++) {
        label_dismension = label_dismension * label_dims_[iter];
    }
    if(data_array[0].size() != label_dismension) {
        std::cerr<<"The dimension of label is error..."<<"("<<data_array[0].size()<<" vs "<<label_dismension<<")"<<std::endl;
        return;
    }
    //extend data set first
    label_dims_[0] = label_dims_[0] + hsize_t(data_array.size());
    dataset_label_->extend(&label_dims_[0]);
    //extend space
    H5::DataSpace *filespace = new H5::DataSpace(dataset_label_->getSpace());
    int num_axes = label_dims_.size();
    std::vector<hsize_t> dims_ext(num_axes, 0);
    for(int iter = 0;iter<num_axes;iter++ ) {
        dims_ext[iter] = label_dims_[iter];
    }
    dims_ext[0] = hsize_t(data_array.size());
    std::vector<hsize_t> offset(num_axes, 0);
    offset[0] = label_offset_;
    filespace->selectHyperslab(H5S_SELECT_SET, &dims_ext[0], &offset[0]);
    //create new data space
    H5::DataSpace *memspace = new H5::DataSpace(num_axes, &dims_ext[0], NULL);
    //write the data into data set
    int data_ext_num = 1;
    for(int iter = 0;iter<num_axes;iter++ ) {
        data_ext_num *= dims_ext[iter];
    }
    if (data_array.size()*data_array[0].size() != data_ext_num) {
        std::cerr<<"Wrong dimension of label..."<<std::endl;
        return;
    }
    float *data_ext = new float[data_ext_num];
    for(int iter = 0; iter < int(data_array.size()); iter++) {
        for(int jter = 0; jter < int(data_array[0].size()); jter++) {
            data_ext[iter*label_dismension + jter] = data_array[iter][jter];
        }
    }

    dataset_label_->write(data_ext, label_type_, *memspace, *filespace);
    label_offset_ = label_offset_ + hsize_t(data_array.size());
    //free memories
    delete [] data_ext;
    delete memspace;
    delete filespace;
}

/*
 * Transfer the std::vector<cv::Mat> into float32 array
 * prepared the data for writing into HDF5 file.
 */
void Mat2H5::transfer2array(float *dst_array, const std::vector<cv::Mat> &sources) {
    int num_patch = sources.size();
    int num_channels = sources[0].channels();
    int height = sources[0].rows;
    int width = sources[0].cols;
//    std::printf("h:%d, w:%d, c:%d, n:%d\n", height, width, num_channels, num_patch);
//    std::printf("The size of sources is %d(m: %d, v: %d)\n", sources.size(), mean_value_.size(),  shrink_ratio_.size());
    for(int iter_n = 0; iter_n<num_patch; iter_n++) {
        cv::Mat img_tmp;
        sources[iter_n].convertTo(img_tmp, CV_32FC1);
//        std::cout<<iter_n<<": The size of image is "<<img_tmp.size()<<std::endl;
        for(int iter_c = 0; iter_c<num_channels ; iter_c++) {
            for(int iter_h = 0; iter_h<height; iter_h++) {
                for(int iter_w = 0; iter_w<width; iter_w++) {
                    int index = ( (iter_n*num_channels + iter_c)*height + iter_h)*width + iter_w;
                    float elem = 0;
                    if(img_tmp.channels() != 2) {
                        elem = (img_tmp.at<float>(iter_h, iter_w, iter_c) - mean_value_[iter_c]) *
                               shrink_ratio_[iter_c];
                    }
                    else if(img_tmp.channels() == 2) {
                        elem = (img_tmp.at<cv::Vec2f>(iter_h, iter_w)[iter_c] - mean_value_[iter_c]) *
                                     shrink_ratio_[iter_c];
                    }
//                    std::printf("(%d %d %d,", iter_h, iter_w, iter_c);
//                    float elem = img_tmp.at<cv::Vec2f>(iter_h, iter_w)[iter_c];
                    *(dst_array + index) = elem;
//                    if(iter_c == 0) std::printf("%f)  ", elem);
                }
//                if(iter_c == 0) std::printf("--\n");
            }
//            if(iter_c == 0) std::printf("=====================================================================================\n");
        }
    }
//    std::printf("Convert Completed...\n");
}

H5::PredType Mat2H5::get_type(std::string data_type) {
    if(boost::iequals(data_type, "float")) return H5::PredType::NATIVE_FLOAT;
    if(boost::iequals(data_type, "int")) return H5::PredType::NATIVE_INT;
    if(boost::iequals(data_type, "uint8")) return H5::PredType::NATIVE_UINT8;
    if(boost::iequals(data_type, "uint16")) return H5::PredType::NATIVE_UINT16;
    if(boost::iequals(data_type, "double")) return H5::PredType::NATIVE_DOUBLE;
    if(boost::iequals(data_type, "uint32")) return H5::PredType::NATIVE_UINT32;
    if(boost::iequals(data_type, "uchar")) return H5::PredType::NATIVE_UCHAR;
}

void Mat2H5::write_tuples(const cv::Mat & img, const std::vector<float> &label) {
    //Fisrt of all, write the data into buffers
    cv::Mat img_tr = img.clone();
    cv::transpose(img_tr, img_tr);
    data_buffer_[buffer_index_] = img_tr;
    label_buffer_[buffer_index_] = label;
    buffer_index_ ++;
    //If the buffer is full, then write the into hdf5 file at once
    if(buffer_index_ == chunk_size_) {
        write_data2hdf5(data_buffer_);
        write_label2hdf5(label_buffer_);
        //Reset the buffers
        data_buffer_.clear();
        data_buffer_.resize(chunk_size_);
        label_buffer_.clear();
        label_buffer_.resize(chunk_size_);
        buffer_index_ = 0;
    }
}
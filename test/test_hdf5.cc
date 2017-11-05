#include <opencv2/opencv.hpp>
#include "hdf5_ct.h"

int main(int argc, char * argv[]) {
    boost::shared_ptr<Mat2H5> transfer;
    transfer = boost::make_shared<Mat2H5>(17.2196f, 0.0125f);
    transfer->create_hdf5("test.h5");
    for(int iter = 0; iter < 1000; iter++) {
        Mat2H5::h5dataset data_sets;
        data_sets.datasetname = "data";
        data_sets.rank = 4;
        data_sets.dimension.push_back(2);//num of batches
        data_sets.dimension.push_back(1);//channels
        data_sets.dimension.push_back(24);//height
        data_sets.dimension.push_back(12);//width
        data_sets.datatype = "float";
        transfer->open_hdf5("test.h5");
        cv::Mat image;
        image = cv::imread("../matlab/samples/test_hdf5.png", CV_8UC1);
        cv::transpose(image, image);
        std::vector<cv::Mat> image_set;
        image_set.push_back(image);
        image_set.push_back(image);
        std::cout << "The size of image is " << image.rows << std::endl;
        transfer->write_mat2hdf5(data_sets, image_set);
        Mat2H5::h5dataset data_label;
        data_label.datasetname = "label";
        data_label.rank = 2;
        data_label.dimension.push_back(2);
        data_label.dimension.push_back(4);
        data_label.datatype = "float";
        float label[] = {1.0f, 3.0f, 2.0f, 4.0f, 9.0f, 8.0f, 6.0f, 7.0f};
        transfer->write_array2hdf5(data_label, label);
        transfer->close_hdf5();
    }
    return 0;
}
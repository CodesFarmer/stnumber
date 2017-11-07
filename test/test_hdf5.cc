#include <opencv2/opencv.hpp>
#include "hdf5_ct.h"

int main(int argc, char * argv[]) {
    std::vector<int> data_dimension;
    data_dimension.push_back(2);//num of batches
    data_dimension.push_back(1);//channels
    data_dimension.push_back(24);//height
    data_dimension.push_back(12);//width

    std::vector<int> label_dimension;
    label_dimension.push_back(2);
    label_dimension.push_back(4);

    Mat2H5 *transfer = new Mat2H5(17.2196f, 0.0125f, 2);
    transfer->create_hdf5("test.h5");
    std::cout<<"Create Success! 1"<<std::endl;
    transfer->create_dataset(Mat2H5::DATA, data_dimension, "float");
    std::cout<<"Create Success! 2"<<std::endl;
    transfer->create_dataset(Mat2H5::LABEL, label_dimension, "float");
    for(int iter = 0; iter < 1000; iter++) {
        transfer->open_hdf5("test.h5");
        cv::Mat image;
        image = cv::imread("../matlab/samples/test_hdf5.png", CV_8UC1);
        cv::transpose(image, image);
        std::vector<cv::Mat> image_set;
        image_set.push_back(image);
        image_set.push_back(image);
        std::cout << "The size of image is " << image.rows << std::endl;
        transfer->write_data2hdf5(image_set);
        float label[] = {1.0f, 3.0f, 2.0f, 4.0f, 9.0f, 8.0f, 6.0f, 7.0f};
        transfer->write_label2hdf5(label, 2);
        transfer->close_hdf5();
    }
    return 0;
}
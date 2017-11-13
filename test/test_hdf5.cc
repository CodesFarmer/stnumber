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

    std::vector<float> mean_value(1, 17.2196f);
    std::vector<float> variance(1, 0.0125f);
    Mat2H5 *transfer = new Mat2H5(mean_value, variance, 100);
    transfer->create_hdf5("test.h5");
    std::cout<<"Create Success! 1"<<std::endl;
    transfer->create_dataset(Mat2H5::DATA, data_dimension, "float");
    std::cout<<"Create Success! 2"<<std::endl;
    transfer->create_dataset(Mat2H5::LABEL, label_dimension, "float");
//    transfer->open_hdf5("test.h5");
    for(int iter = 0; iter < 1000; iter++) {
        cv::Mat image;
        image = cv::imread("../matlab/samples/test_hdf5.png", CV_8UC1);
        float label[] = {9.6f, 8.1f, 6.2f, 7.5f};
        std::vector<float> label_cp(&label[0], &label[0] + 4);
        transfer->write_tuples(image, label_cp);
    }
    transfer->close_hdf5();
    return 0;
}
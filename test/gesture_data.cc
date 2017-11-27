//
// Created by slam on 17-11-27.
//

#include "hdf5_ct.h"
#include <ctime>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    std::vector<int> data_dimension;
    data_dimension.push_back(1);//num of batches
    data_dimension.push_back(1);//channels
    data_dimension.push_back(128);//height
    data_dimension.push_back(128);//width

    std::vector<int> label_dimension;
    label_dimension.push_back(1);
    label_dimension.push_back(1);
    std::vector<float> mean_value(1, 0.0f);
    std::vector<float> variance(1, 0.0125f);
    Mat2H5 *transfer = new Mat2H5(mean_value, variance, 100);
    transfer->create_hdf5(argv[2]);
    transfer->create_dataset(Mat2H5::DATA, data_dimension, "float");
    transfer->create_dataset(Mat2H5::LABEL, label_dimension, "float");

    std::ifstream input_fid;
    input_fid.open(argv[1], std::ios::in);
    std::string file_path;
    float label;
    srand(time(NULL));
    while(!input_fid.eof()) {
        //Read the image and write them into hdf5
        input_fid >> file_path;
        input_fid >> label;
        cv::Mat image = cv::imread(file_path, CV_8UC1);
        transfer->write_tuples(image, std::vector<float>(1, label));
        //select 2/3 data for augmentation
        int rand_num = rand()%3;
        std::cout<<image.size()<<" "<<label<<std::endl;
        if(rand_num != 0) {
            //transpose the image
            cv::Mat image_tr;
            cv::transpose(image, image_tr);
            transfer->write_tuples(image_tr, std::vector<float>(1, label));
            //flip the image
            cv::Mat image_fp;
            cv::flip(image, image_fp, 0);
            transfer->write_tuples(image_fp, std::vector<float>(1, label));
            //flip and transpose the image
            cv::Mat image_tf;
            cv::transpose(image, image_tf);
            cv::flip(image_tf, image_tf, 1);
            transfer->write_tuples(image_tf, std::vector<float>(1, label));
        }
    }
    input_fid.close();
    transfer->close_hdf5();
    return 0;
}
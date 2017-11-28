//
// Created by slam on 17-11-27.
//

#include "hdf5_ct.h"
#include <ctime>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

#define _CROP_NUM 10
#define _PAD_PIXEL 16

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
//        std::cout<<image.size()<<" "<<label<<std::endl;
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
            //if the image is fist, we crop it and augment it
            if(label == 2) {
//                int is_crop = rand()%2;
//                if(is_crop == 0) {
                    //at first, we padding the image with 16 pixels at each edge
                    cv::Mat image_pad(image.rows + _PAD_PIXEL*2, image.cols + _PAD_PIXEL*2, CV_8UC1);
                    cv::randu(image_pad, cv::Scalar(0,0,0), cv::Scalar(100,100,100));
                    cv::Rect ori_bbx(_PAD_PIXEL, _PAD_PIXEL, image.rows, image.cols);
                    image_pad(ori_bbx) = image;
                    int iter = 0;
                    while(iter < _CROP_NUM) {
                        //We generate a number for crop the images
                        int shift_offset_x = rand()%33 - 16;
                        int shift_offset_y = rand()%33 - 16;
                        if(std::abs(shift_offset_x) > 4 && std::abs(shift_offset_y) > 4) {
                            cv::Rect crop_bbx(_PAD_PIXEL + shift_offset_x, _PAD_PIXEL + shift_offset_y, image.rows, image.cols);
                            cv::Mat image_crop = image_pad(crop_bbx).clone();
                            transfer->write_tuples(image_crop, std::vector<float>(1, label));
                            iter++;
                        }
                    }
//                }
            }
        }
    }
    input_fid.close();
    transfer->close_hdf5();
    return 0;
}
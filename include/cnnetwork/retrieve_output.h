//
// Created by lowell on 10/27/17.
//
/*
 * This class is not finished yet...
 * And the function is not test...
 * */

#ifndef PROJECT_RETRIEVE_OUTPUT_H_H
#define PROJECT_RETRIEVE_OUTPUT_H_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/memory_data_layer.hpp"

typedef float Dtype;

class CNNTools {
public:
    CNNTools(std::string prototxt_path, std::string model_path, float img_norm, const std::vector<Dtype>& mean_value) {
        initialize_network(prototxt_path, model_path);
        initialize_transform(img_norm, mean_value);
    }

    void feed_network_imageset(std::vector<cv::Mat> &imgs) {
        for(int iter = 0; iter < int(imgs.size()); iter++) {
            feed_network_image(imgs[iter]);
        }
    }

    void feed_network_image(cv::Mat &img) {
        cv::Mat img_bk;
        img_bk = img.clone();
        caffe::Blob<Dtype> input_blob;
        input_blob.Reshape(1, input_channels_, input_height_, input_width_);
        cnn_transform_->Transform(img_bk, &input_blob);
        std::vector<caffe::Blob<Dtype> *> input_data;
        input_data.push_back(&input_blob);

        int layer_id = 0;
        int top_id = neural_network_->top_ids(layer_id)[0];
        std::vector<caffe::Blob<Dtype>*> top_vec = neural_network_->top_vecs()[top_id];
        caffe::Blob<Dtype>* blob_ptr = top_vec[0];
        blob_ptr->Reshape(1, input_channels_, input_height_, input_width_);

        neural_network_->Forward(input_data);
    }

    //put the matrix of opencv into neural network and pass forward
    void feed_network_matrix(cv::Mat &data_matrix){
        caffe::Blob<Dtype> input_blob;
        input_blob.Reshape(1, input_channels_, input_height_, input_width_);
        std::vector<cv::Mat> data_matrix_4d;
        data_matrix_4d.push_back(data_matrix);
        mat_into_blob(data_matrix_4d, &input_blob);
        std::vector<caffe::Blob<Dtype> *> input_data;
        input_data.push_back(&input_blob);

        int layer_id = 0;
        int top_id = neural_network_->top_ids(layer_id)[0];
        std::vector<caffe::Blob<Dtype>*> top_vec = neural_network_->top_vecs()[top_id];
        caffe::Blob<Dtype>* blob_ptr = top_vec[0];
        blob_ptr->Reshape(1, input_channels_, input_height_, input_width_);

        neural_network_->Forward(input_data);
    }
    void output_by_name(std::string blob_name, std::vector<Dtype>& the_vec){
        boost::shared_ptr<caffe::Blob<Dtype> > the_blob;
        the_blob = neural_network_->blob_by_name(blob_name);
        blob_into_vec(the_blob, the_vec);
    }

private:
    void mat_into_blob(std::vector<cv::Mat> & data_matrix, caffe::Blob<Dtype>* the_blob) {
        int num_patch = the_blob->num();
        int num_channels = the_blob->channels();
        int height = the_blob->height();
        int width = the_blob->width();
        if(height != data_matrix[0].rows && width != data_matrix[0].cols) {
            std::cerr<<"The size of Blob is (H: "<<height<<", W: "<<width<<")"<<std::endl;
            return;
        }

        int data_num = num_patch*num_channels*height*width;
        Dtype * data_vec = new Dtype[data_num];
        for(int iter_p = 0; iter_p < num_patch; iter_p++) {
            cv::Mat cur_mat = data_matrix[iter_p];
            for(int iter_c = 0; iter_c < num_channels; iter_c++) {
                for(int iter_h = 0; iter_h < height; iter_h++) {
                    for(int iter_w = 0; iter_w < width; iter_w ++) {
                        Dtype the_value;
                        the_value = cur_mat.at<float>(iter_h, iter_w, iter_c);
                        int index = ( (iter_p*num_channels + iter_c)*height + iter_h)*width + iter_w;
                        *(data_vec + index) = the_value;
                    }
                }
            }
        }
        the_blob->set_cpu_data(data_vec);
    }

    void blob_into_vec(boost::shared_ptr<caffe::Blob<Dtype> > the_blob,
                       std::vector<Dtype> &the_vec) {
        int num_patch = the_blob->num();
        int num_channels = the_blob->channels();
        int height = the_blob->height();
        int width = the_blob->width();
        the_vec.clear();

        const Dtype *data_ptr = the_blob->cpu_data();
        for(int iter_p = 0; iter_p<num_patch; iter_p++) {
            for(int iter_c = 0; iter_c<num_channels; iter_c++) {
                for(int iter_h = 0; iter_h < height; iter_h++) {
                    for(int iter_w = 0; iter_w < width; iter_w ++) {
                        Dtype the_value;
                        int index = ( (iter_p*num_channels + iter_c)*height + iter_h)*width + iter_w;
                        the_value = *(data_ptr + index);
                        the_vec.push_back(the_value);
                    }
                }
            }
        }
    }
    int initialize_network(std::string prototxt_path, std::string model_path) {
        if(access(prototxt_path.c_str(), F_OK) == -1) {
            std::cerr<<"The file "<<prototxt_path<<" does not exist!"<<std::endl;
            return -1;
        }
        if(access(model_path.c_str(), F_OK) == -1) {
            std::cerr<<"The file "<<prototxt_path<<" does not exist!"<<std::endl;
            return -2;
        }
        try{
            boost::shared_ptr<caffe::Net<Dtype> > tmp_network(new caffe::Net<Dtype>(prototxt_path, caffe::TEST, 0, NULL));
            neural_network_ = tmp_network;
        }
        catch(int e){
            std::cerr<<"Can not open the network proto "<<prototxt_path<<std::endl;
            return -1;
        }
        try{
            neural_network_->CopyTrainedLayersFrom(model_path);
            int layer_id = 0;
            int top_id = neural_network_->top_ids(layer_id)[0];
            std::vector<caffe::Blob<Dtype>*> top_vec = neural_network_->top_vecs()[top_id];
            caffe::Blob<Dtype>* blob_ptr = top_vec[0];
            input_patch_ = blob_ptr->num();
            input_channels_ = blob_ptr->channels();
            input_height_ = blob_ptr->height();
            input_width_ = blob_ptr->width();
        }
        catch(int e){
            std::cerr<<"Can not load the pre-trained model "<<model_path<<std::endl;
            return -2;
        }
        return 0;
    }
    int initialize_transform(float img_norm, const std::vector<Dtype>& mean_value) {
        caffe::TransformationParameter transform_params;
        transform_params.set_scale(img_norm);
        for(size_t iter = 0;iter<mean_value.size();iter++){
            transform_params.add_mean_value(mean_value[iter]);
        }
        boost::shared_ptr<caffe::DataTransformer<Dtype> > tmp_transform(new caffe::DataTransformer<Dtype>(transform_params, caffe::TEST));
        cnn_transform_ = tmp_transform;
    }

private:
    boost::shared_ptr<caffe::Net<Dtype> > neural_network_;
    boost::shared_ptr<caffe::DataTransformer<Dtype> > cnn_transform_;
    int input_height_;
    int input_width_;
    int input_patch_;
    int input_channels_;
};

#endif //PROJECT_RETRIVE_OUTPUT_H_H

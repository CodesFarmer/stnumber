#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <vector>

#include "filepath.h"
#include "readxml_ct.h"
#include "detect_face.h"

int main(int argc, char * argv[]) {
    std::string image_path(argv[1]);
    cv::Mat image = cv::imread(image_path, CV_8UC1);
    std::map<std::string, std::pair<std::string, std::string> > modelpath;
    std::string models_dir = "../data/model/";
    modelpath["pnet"] = std::make_pair(models_dir + std::string("pnet.caffemodel"), models_dir+std::string("pnet_deploy.prototxt"));
    modelpath["rnet"] = std::make_pair(models_dir + std::string("rnet.caffemodel"), models_dir+std::string("rnet_deploy.prototxt"));
    std::vector<float> mean_value(1, 17.2196);
    float img2net_scale = 0.0125;
    FaceDetector<float>* detector = new FaceDetector<float>();
    if(detector->initialize_network(modelpath)<0) {
        return -1;
    }
    detector->initialize_transformer(img2net_scale, mean_value);
    std::vector<std::vector<float> > faces_bboxes =  detector->detect_face(image);
    std::cout<<faces_bboxes.size()<<std::endl;

    return 0;
}
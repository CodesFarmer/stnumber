//
// Created by slam on 17-11-3.
//

#ifndef PROJECT_HAND_BOUNDINGBOX_H
#define PROJECT_HAND_BOUNDINGBOX_H

#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
//#include "detect_face.h"
//namespace cv {
//    class Rect;
//}
extern "C" {
    int initialize_detector(const std::map<std::string, std::pair<std::string, std::string> > &);
    cv::Rect get_hand_bbx(const cv::Mat &);
};

#endif //PROJECT_HAND_BOUNDINGBOX_H

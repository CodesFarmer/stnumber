//
// Created by slam on 17-10-28.
//

#ifndef PROJECT_READXML_CT_H
#define PROJECT_READXML_CT_H

#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <tinyxml2.h>

std::vector<cv::Rect> get_bounding_boxes(std::string);

#endif //PROJECT_READXML_CT_H

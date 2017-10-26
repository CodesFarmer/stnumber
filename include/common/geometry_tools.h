#ifndef GEOMETRY_TOOLS_H
#define GEOMETRY_TOOLS_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace GEOMETRYTOOLS{
    float regionsIOU(std::vector<cv::Rect>&, cv::Rect&);
    float regionIOU(cv::Rect &, cv::Rect &);
}

#endif
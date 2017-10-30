#include <opencv2/opencv.hpp>
#include <iostream>

#include "retrieve_output.h"
#include "filepath.h"
#include "readxml_ct.h"
#include "geometry_tools.h"

int main(int argv, char * argc[]) {
    std::vector<cv::Rect> bbxes;
    bbxes = get_bounding_boxes("/home/slam/datasets/handdetect_sample/annotations/1509071298190268772.xml");
    cv::Rect rect = bbxes[0];
    cv::Mat img = cv::imread("/home/slam/datasets/handdetect_sample/images/001_01/cam0/1509071298190268772.png", CV_8UC1);

    std::cout<<rect<<std::endl;

    cv::rectangle(img, rect, cv::Scalar(255,255,255), 2);
    cv::imshow("bounding box", img);
    cv::waitKey(10000);

    cv::Rect rects(10, 10, 30, 30);
    cv::Rect the_rect(20, 20, 40, 40);
    float iou = GEOMETRYTOOLS::regionIOU(rects, the_rect);
    return 0;
}
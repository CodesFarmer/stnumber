//
// Created by lowell on 10/25/17.
//

#ifndef PROJECT_GENERATE_PATCH_H
#define PROJECT_GENERATE_PATCH_H

#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <ctime>
#include <sstream>

#include "filepath.h"
#include "geometry_tools.h"

class GeneratePatch{
public:
    GeneratePatch(int num_neg = 50, int num_pos = 25, float neg_iou = 0.3f, float prt_iou = 0.4f, float pos_iou = 0.65f):
            num_negative_(num_neg), num_positive_(num_pos), neg_IOU_(neg_iou), part_IOU_(prt_iou), pos_IOU_(pos_iou){};
    void generatePatches(std::string filename,
                         std::string dst_path);//Similar to above
private:
//    void saveImages(){};
    void createPathces(std::string, std::vector<cv::Rect> &, std::string);
    void createNegativeSamples(cv::Mat &, std::vector<cv::Rect> &, std::string);
    void createPositiveSamples(cv::Mat &, std::vector<cv::Rect> &, std::string, std::string);
private:
    int num_negative_;//How much negative samples we cropping from image
    int num_positive_;//Similar to above
    float neg_IOU_;//The threshold we justify a region is a negative samples
    float part_IOU_;//Similar to above
    float pos_IOU_;//Similar to above
};


#endif //PROJECT_GENERATE_PATCH_H

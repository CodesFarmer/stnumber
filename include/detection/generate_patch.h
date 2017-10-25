//
// Created by lowell on 10/25/17.
//

#ifndef PROJECT_GENERATE_PATCH_H
#define PROJECT_GENERATE_PATCH_H

#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>

class GeneratePatch{
public:
    void generatePatches(std::string,
                         int num_negative = 50,//How much negative samples we cropping from image
                         int num_positive = 25,//Similar to above
                         float neg_IOU = 0.3f,//The threshold we justify a region is a negative samples
                         float part_IOU = 0.4f,//Similar to above
                         float pos_IOU = 0.65f);//Similar to above
private:
    //calculate the ratio of overlap region
    float regionsIOU(cv::Rect & r1, cv::Rect &r2);
    void saveImages();
    void cropImages(std::string img_path, cv::Rect &ojb_bbx, float, float, float);
};


#endif //PROJECT_GENERATE_PATCH_H

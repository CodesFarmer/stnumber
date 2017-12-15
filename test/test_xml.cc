#include <opencv2/opencv.hpp>
#include <fstream>

#include "retrieve_output.h"
#include "filepath.h"
#include "readxml_ct.h"
#include "geometry_tools.h"

int main(int argv, char * argc[]) {
    std::string file_name(argc[1]);
    std::string img_name;
    std::ifstream in_fid;
    in_fid.open(file_name.c_str(), std::ios::in);
    while(!in_fid.eof()) {
        in_fid>>img_name;
        std::vector<cv::Rect> bbxes;
        bbxes = get_bounding_boxes(img_name);
        FILEPARTS::replace_string(img_name, ".xml", ".png");
        FILEPARTS::replace_string(img_name, "xml", "dep0");
        cv::Mat img_16u;
        img_16u = cv::imread(img_name, CV_16UC1);
        cv::bitwise_and(img_16u, 0x1FFF, img_16u);
        cv::Mat hand_depth = img_16u(bbxes[0]).clone();
        std::string im_path;
        std::string im_name;
        std::string im_ext;
        FILEPARTS::fileparts(img_name, im_path, im_name, im_ext);
        std::string dest_path;
        FILEPARTS::fullfile(dest_path, 2, std::string("depth_hand"), im_name+"."+im_ext);
        cv::imwrite(dest_path, hand_depth);
    }


//    std::vector<cv::Rect> bbxes;
//    bbxes = get_bounding_boxes("/home/slam/datasets/handdetect_sample/annotations/1509071298190268772.xml");
//    cv::Rect rect = bbxes[0];
//    cv::Mat img = cv::imread("/home/slam/datasets/handdetect_sample/images/001_01/cam0/1509071298190268772.png", CV_8UC1);
//
//    std::cout<<rect<<std::endl;
//
//    cv::rectangle(img, rect, cv::Scalar(255,255,255), 2);
//    cv::imshow("bounding box", img);
//    cv::waitKey(10000);
//
//    cv::Rect rects(10, 10, 30, 30);
//    cv::Rect the_rect(20, 20, 40, 40);
//    float iou = GEOMETRYTOOLS::regionIOU(rects, the_rect);
    return 0;
}
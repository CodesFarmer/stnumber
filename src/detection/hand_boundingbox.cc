#include "hand_boundingbox.h"

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "detect_face.h"

extern "C" {
boost::shared_ptr<FaceDetector<float> > detector;
int initialize_detector(const std::map<std::string, std::pair<std::string, std::string> > &modelpath) {
    std::vector<float> mean_value(1, 17.2196);
    float img2net_scale = 0.0125;
    detector = boost::make_shared<FaceDetector<float> >();
    if(detector->initialize_network(modelpath)<0) {
        return -1;
    }
    detector->initialize_transformer(img2net_scale, mean_value);
}
//Get the bounding box
cv::Rect get_hand_bbx(const cv::Mat &image) {
    cv::Mat tmpimg = image.clone();
    std::vector<std::vector<float> > hand_bbx = detector->detect_face(tmpimg);
//    std::cout<<"The size of bounding box is "<<hand_bbx.size()<<std::endl;
    cv::Rect hand_rect(-1, -1, -1, -1);
    if(hand_bbx.size() == 0) return hand_rect;
    int x_l = std::max( hand_bbx[0][0], 0.0f );
    int y_l = std::max( hand_bbx[0][1], 0.0f);
    int x_r = std::min( hand_bbx[0][2], float(image.cols-1) );
    int y_r = std::min( hand_bbx[0][3], float(image.rows-1) );
    hand_rect.x = x_l;
    hand_rect.y = y_l;
    hand_rect.width = x_r - x_l + 1;
    hand_rect.height = y_r - y_l + 1;
    return hand_rect;
}
}
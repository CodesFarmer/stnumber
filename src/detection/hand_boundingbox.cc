#include "hand_boundingbox.h"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "detect_face.h"

extern "C" {
boost::shared_ptr<FaceDetector<float> > detector;
std::vector<float> init_rect_;
double t_start = 0.0f;
double t_end = 0.0f;
int initialize_detector(const std::map<std::string, std::pair<std::string, std::string> > &modelpath, int channels) {
//    std::vector<float> mean_value(1, 17.2196);
//    float img2net_scale = 0.0125;
    //Set the channels for preparing data before detection
    std::vector<float> mean_value(channels, 0.0f);
    float img2net_scale = 1.0f;
    detector = boost::make_shared<FaceDetector<float> >(channels);
    if(detector->initialize_network(modelpath)<0) {
        return -1;
    }
    detector->initialize_transformer(img2net_scale, mean_value);
    init_rect_.resize(5, 0);
//    init_rect_.first = 0;
//    init_rect_.second.x = 0;
//    init_rect_.second.y = 0;
//    init_rect_.second.width = 0;
//    init_rect_.second.height = 0;
}
//Get the bounding box
cv::Rect get_hand_bbx(const cv::Mat &image) {
    //pre-processing
    float mean_ir = 17.2196f;
    float scale_ir = 0.0125f;
    cv::Mat tmpimg;
    image.convertTo(tmpimg, CV_32FC1);
    tmpimg = (tmpimg - mean_ir)*scale_ir;

    std::vector<std::vector<float> > hand_bbx = detector->tracking_hand(tmpimg, init_rect_, 0.95);
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
    hand_bbx[0][0] = x_l;
    hand_bbx[0][1] = y_l;
    hand_bbx[0][2] = x_r;
    hand_bbx[0][3] = y_r;
    init_rect_ = hand_bbx[0];
    return hand_rect;
}

//Get the bounding box
cv::Rect get_hand_bbx_irdp(const cv::Mat &img_ir, const cv::Mat &img_dp) {
    //Preprocessment
    cv::Mat image;
    pre_processing(img_ir, img_dp, image);

    //Reset the probability of previous bounding box
    t_end = (double)cvGetTickCount();
    if((t_end - t_start)/(cvGetTickFrequency()*500)  > 500.0f) {
        std::printf("TEST\n");
        init_rect_[4] = 0.0f;
        t_start = t_end;
    }
    //Forward pass the neural network and get the bounding boxes
    std::vector<std::vector<float> > hand_bbx = detector->tracking_hand(image, init_rect_, 0.8);
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
    hand_bbx[0][0] = x_l;
    hand_bbx[0][1] = y_l;
    hand_bbx[0][2] = x_r;
    hand_bbx[0][3] = y_r;
    init_rect_ = hand_bbx[0];
    return hand_rect;
}

bool pre_processing(const cv::Mat &img_ir, const cv::Mat &img_dp, cv::Mat & output) {
    float mean_ir = 0.0f;
    float mean_dp = 0.0f;
    float scale_ir = 0.0125f;
    float scale_dp = 0.00083f;
    cv::Mat tmpimg_ir = img_ir.clone();
    cv::Mat tmpimg_dp = img_dp.clone();
    cv::bitwise_and(tmpimg_dp, 0x1FFF, tmpimg_dp);
    cv::Mat img_ir_float(tmpimg_ir.size(), CV_32FC1);
    cv::Mat img_dp_float(tmpimg_dp.size(), CV_32FC1);
    tmpimg_ir.convertTo(img_ir_float, CV_32FC1);
    img_ir_float = (img_ir_float - mean_ir)*scale_ir;
    tmpimg_dp.convertTo(img_dp_float, CV_32FC1);
    img_dp_float = (img_dp_float - mean_dp)*scale_dp;
    std::vector<cv::Mat> image_ir_dp;
    image_ir_dp.push_back(img_ir_float);
    image_ir_dp.push_back(img_dp_float);
//    output(tmpimg_ir.size(), CV_32FC2);
    cv::merge(image_ir_dp, output);
}
}
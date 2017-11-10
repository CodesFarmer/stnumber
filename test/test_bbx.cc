#include "hand_boundingbox.h"

int main(int argc, char * argv[]) {
    std::string imgpath(argv[1]);
    std::map<std::string, std::pair<std::string, std::string> > modelpath;
    std::string models_dir = "../data/model/";
    modelpath["pnet"] = std::make_pair(models_dir + std::string("pnet_iter_200000.caffemodel"), models_dir+std::string("pnet_deploy.prototxt"));
    modelpath["rnet"] = std::make_pair(models_dir + std::string("rnet_iter_20000.caffemodel"), models_dir+std::string("rnet_deploy.prototxt"));
    modelpath["onet"] = std::make_pair(models_dir + std::string("onet_iter_10000.caffemodel"), models_dir+std::string("onet_deploy.prototxt"));
    initialize_detector(modelpath);
    cv::Mat image;
    image = cv::imread(imgpath, CV_8UC1);
//    cv::imshow("test", image);
//    cv::waitKey(0);
    cv::Rect hand_bbx = get_hand_bbx(image);
    cv::rectangle(image, hand_bbx, cv::Scalar(255));
    cv::imshow("BBX", image);
    cv::waitKey(0);
}
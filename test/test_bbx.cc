#include "hand_boundingbox.h"

int main(int argc, char * argv[]) {
    std::map<std::string, std::pair<std::string, std::string> > modelpath;
    std::string models_dir = "../data/model/";
    modelpath["pnet"] = std::make_pair(models_dir + std::string("pnet.caffemodel"), models_dir+std::string("pnet_deploy.prototxt"));
    modelpath["rnet"] = std::make_pair(models_dir + std::string("rnet.caffemodel"), models_dir+std::string("rnet_deploy.prototxt"));
    modelpath["onet"] = std::make_pair(models_dir + std::string("onet.caffemodel"), models_dir+std::string("onet_deploy.prototxt"));
    initialize_detector(modelpath);
    cv::Mat image;
    image = cv::imread("test.png", CV_8UC1);
    cv::Rect hand_bbx = get_hand_bbx(image);
}
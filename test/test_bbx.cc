#define GOOGLE_STRIP_LOG 0
#include "hand_boundingbox.h"
#include <fstream>

int main(int argc, char * argv[]) {
    std::freopen("/ff/null", "w", stderr);
    std::freopen("/ff/null", "w", stdout);

    std::string imgs_path(argv[1]);
    std::map<std::string, std::pair<std::string, std::string> > modelpath;
    std::string models_dir = "../data/model/";
    modelpath["pnet"] = std::make_pair(models_dir + std::string("pnet.caffemodel"), models_dir+std::string("pnet_deploy.prototxt"));
    modelpath["rnet"] = std::make_pair(models_dir + std::string("rnet.caffemodel"), models_dir+std::string("rnet_deploy.prototxt"));
    modelpath["onet"] = std::make_pair(models_dir + std::string("onet.caffemodel"), models_dir+std::string("onet_deploy.prototxt"));
    initialize_detector(modelpath);

    std::ifstream input_fid;
    input_fid.open(imgs_path.c_str(), std::ios::in);
    std::string img_name;

    while(!input_fid.eof()) {
        cv::Mat image;
        input_fid>>img_name;
        image = cv::imread(img_name, CV_8UC1);
//    cv::imshow("test", image);
//    cv::waitKey(0);
        cv::Rect hand_bbx;
        hand_bbx = get_hand_bbx(image);
        cv::rectangle(image, hand_bbx, cv::Scalar(255));
        cv::imshow("BBX", image);
        cv::waitKey(0);
    }
}
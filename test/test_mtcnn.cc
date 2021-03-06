#include <opencv2/opencv.hpp>
#include <ctime>
#include <sys/time.h>
#include <iomanip>

#include "filepath.h"
#include "readxml_ct.h"
#include "detect_face.h"
#include "hand_boundingbox.h"

int main(int argc, char * argv[]) {
    std::string imgs_path(argv[1]);
    std::map<std::string, std::pair<std::string, std::string> > modelpath;
    std::string models_dir = "../data/model/";
    bool ironly = false;
    bool isrotate = false;
    if(!ironly) {
        //Depth and Infrared images both
        modelpath["pnet"] = std::make_pair(models_dir + std::string("pnet_dpir.caffemodel"),
                                           models_dir + std::string("pnet_deploy.prototxt"));
        modelpath["rnet"] = std::make_pair(models_dir + std::string("rnet_dpir.caffemodel"),
                                           models_dir + std::string("rnet_deploy.prototxt"));
        modelpath["onet"] = std::make_pair(models_dir + std::string("onet_dpir.caffemodel"),
                                           models_dir + std::string("onet_deploy.prototxt"));
//        modelpath["tnet"] = std::make_pair(models_dir + std::string("tnet.caffemodel"),
//                                           models_dir + std::string("tnet_deploy.prototxt"));
    }
    else if(ironly){//IR only
        modelpath["pnet"] = std::make_pair(models_dir + std::string("pnet_1.caffemodel"), models_dir+std::string("pnet_deploy.prototxt"));
        modelpath["rnet"] = std::make_pair(models_dir + std::string("rnet_1.caffemodel"), models_dir+std::string("rnet_deploy.prototxt"));
        modelpath["onet"] = std::make_pair(models_dir + std::string("onet_1.caffemodel"), models_dir+std::string("onet_deploy.prototxt"));
    }
    std::ifstream input_fid;
    input_fid.open(imgs_path.c_str(), std::ios::in);
    std::string img_name;

    struct timeval formertime;
    struct timeval curtime;
    gettimeofday(&formertime, NULL);
    int jter = 0;
    double sum_time = 0.0;
    if(!ironly) initialize_detector(modelpath, 2);
    else initialize_detector(modelpath, 1);
    int testnum = 0;
    double time_start = (double)cvGetTickCount();
    double time_end = 0.0;
    while(!input_fid.eof() && jter <1000) {
        input_fid>>img_name;
        time_end = (double)cvGetTickCount();
        while((time_end - time_start)/(cvGetTickFrequency()*1000) < 50.0f) time_end = (double)cvGetTickCount();
        time_start = time_end;
        cv::Mat image_ir = cv::imread(img_name, CV_8UC1);
        if(image_ir.empty()) continue;
        FILEPARTS::replace_string(img_name, "cam0", "dep0");
        cv::Mat image_dp = cv::imread(img_name, CV_16UC1);
        if(image_dp.empty()) continue;
        cv::bitwise_and(image_dp, 0x1FFF, image_dp);
        if(isrotate) {
            cv::transpose(image_ir, image_ir);
            cv::transpose(image_dp, image_dp);
        }

        gettimeofday(&formertime, NULL);
//        cv::Rect hand_bbx;
        cv::Rect hand_bbx;
        if(!ironly) hand_bbx = get_hand_bbx_irdp(image_ir, image_dp);
        else hand_bbx = get_hand_bbx(image_ir);
//        detector->detect_face(image);
        gettimeofday(&curtime, NULL);
        double time_cost = (curtime.tv_sec - formertime.tv_sec) + (curtime.tv_usec - formertime.tv_usec) / 1000000.0;
        sum_time += time_cost;
        std::cout<<time_cost<<std::endl;
        jter++;

        //Display the image
        cv::rectangle(image_ir, hand_bbx, cv::Scalar(255));
        if(isrotate) cv::transpose(image_ir, image_ir);
//        cv::imshow("BBX", image_ir);
//        cv::waitKey(0);
        //Write the image
        std::string file_path;
        std::string file_name;
        std::string file_ext;
        FILEPARTS::fileparts(img_name, file_path, file_name, file_ext);
        std::string dest_path;
        if(!ironly) FILEPARTS::fullfile(dest_path, 2, std::string("data_onet"), file_name+"."+file_ext);
        else FILEPARTS::fullfile(dest_path, 2, std::string("data_ir"), file_name+"."+file_ext);
//        std::cout<<dest_path<<std::endl;
        cv::imwrite(dest_path, image_ir);
        testnum++;
    }
    std::cout<<"The mean time cost: "<<sum_time/float(testnum)<<std::endl;

    return 0;
}
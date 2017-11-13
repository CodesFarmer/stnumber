//
// Created by lowell on 10/25/17.
//

#ifndef PROJECT_GENERATE_PATCH_H
#define PROJECT_GENERATE_PATCH_H

#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <sstream>
#include <iomanip>
#include <unistd.h>
#include <ctime>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "filepath.h"
#include "readxml_ct.h"
#include "geometry_tools.h"
#include "detect_face.h"
#include "hdf5_ct.h"

class GeneratePatch{
public:
    enum SaveMode {
        DISK = 3,
        HDF5 = 4
    };
public:
    GeneratePatch(int num_neg = 50, int num_pos = 25, float neg_iou = 0.3f, float prt_iou = 0.4f, float pos_iou = 0.65f):
            num_negative_(num_neg), num_positive_(num_pos), neg_IOU_(neg_iou), part_IOU_(prt_iou), pos_IOU_(pos_iou){};
    void generate_patches_disk(std::string filename,
                         int img_size,
                         std::string dst_path);//Similar to above
    void generate_patches_hdf5(std::string filename,
                               int img_size,
                               std::string dst_path);//Similar to above
    void generate_patches_cnn(std::string, int, std::string);
    void initialize_detector(const std::map<std::string, std::pair<std::string, std::string> > &, const float, const std::vector<float>);
private:
    void create_patches(std::string, std::vector<cv::Rect> &, int, std::string);
    void create_negative_samples(cv::Mat &, std::vector<cv::Rect> &, int, std::string);
    void create_positive_samples(cv::Mat &, std::vector<cv::Rect> &, int, std::string, std::string);
    void write_to_disk(const cv::Mat & image, int img_size, const std::string & name_prefix, int label,
                       const bool augmentation, const cv::Rect &, const cv::Rect &);
    bool transfer_16u28u(cv::Mat &img_16u, cv::Mat &img_8u);
    void save2hdf5(const cv::Mat & image, const std::vector<float> label);
    void save2disk(const cv::Mat & image, const std::vector<float> label, const std::string &);
private:
    int num_negative_;//How much negative samples we cropping from image
    int num_positive_;//Similar to above
    float neg_IOU_;//The threshold we justify a region is a negative samples
    float part_IOU_;//Similar to above
    float pos_IOU_;//Similar to above
    std::ofstream negative_fid_;
    std::ofstream positive_fid_;
    std::ofstream part_fid_;
    boost::shared_ptr<FaceDetector<float> > detector_;
    boost::shared_ptr<Mat2H5> hdf5_writer_;
    SaveMode save_mode_;
};


#endif //PROJECT_GENERATE_PATCH_H

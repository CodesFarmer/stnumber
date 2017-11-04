#include <opencv2/opencv.hpp>
#include "hdf5_ct.h"

int main(int argc, char * argv[]) {
    CTHDF5::h5dataset data_sets;
    data_sets.datasetname = "hello";
    data_sets.rank = 2;
    data_sets.dimension.push_back(4);
    data_sets.dimension.push_back(2);
    data_sets.datatype = "float";
    CTHDF5::create_hdf5("test.h5");
    hid_t hdf5_fid = CTHDF5::open_hdf5("test.h5");
    cv::Mat image;
    image = cv::imread("../data/samples/101.png");
    std::vector<cv::Mat> image_set;
    image_set.push_back(image);
    write2hdf5(hdf5_fid, data_sets, image_set);
    CTHDF5::close_hdf5(hdf5_fid);
    return 0;
}
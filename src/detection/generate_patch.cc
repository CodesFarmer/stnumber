#include "generate_patch.h"

void GeneratePatch::generatePatches(std::string filename,
                                    int num_negative,
                                    int num_positive,
                                    float neg_IOU,
                                    float part_IOU,
                                    float pos_IOU) {
    std::ifstream input_fid;
    input_fid.open(filename.c_str());

    std::string img_path;
    float points;
    cv::Rect object_bbx;
    while(!input_fid.eof()) {
        input_fid>>img_path;
        input_fid>>points;
        object_bbx.x = points;
        input_fid>>points;
        object_bbx.y = points;
        input_fid>>points;
        object_bbx.width = points - object_bbx.x + 1.0f;
        input_fid>>points;
        object_bbx.height = points - object_bbx.y + 1.0f;
        cropImages(img_path, object_bbx, neg_IOU, part_IOU, pos_IOU);
    }
}

void Generate::cropImages(std::string img_path, cv::Rect ojb_bbx, float neg_IOU, float part_IOU, float pos_IOU, std::string dest_path) {
    cv::Mat image = cv::imread(img_path);
}

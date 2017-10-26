#include "generate_patch.h"

void GeneratePatch::generatePatches(std::string filename, std::string dst_path) {
    std::ifstream input_fid;
    input_fid.open(filename.c_str());

    std::string img_path;
    std::string all_bbxes;
    float points;
    std::vector<cv::Rect> bounding_boxes;
    while(!input_fid.eof()) {
        input_fid>>img_path;
        std::getline(input_fid, all_bbxes);
        std::istringstream input_string(all_bbxes);
        while(input_string>>points) {
            cv::Rect object_bbx;
            object_bbx.x = points;
            input_string >> points;
            object_bbx.y = points;
            input_string >> points;
            object_bbx.width = points - object_bbx.x + 1.0f;
            input_string >> points;
            object_bbx.height = points - object_bbx.y + 1.0f;
            bounding_boxes.push_back(object_bbx);
        }
        createPathces(img_path, bounding_boxes, dst_path);
    }
}

void GeneratePatch::createPathces(std::string img_path, std::vector<cv::Rect> &ojb_bbxes, std::string dest_path) {
    cv::Mat image = cv::imread(img_path);
    //get the details of file path
    std::string file_path;
    std::string file_name;
    std::string file_ext;
    FILEPARTS::fileparts(img_path, file_path, file_name, file_ext);

    std::string negative_path = dest_path + "/negative/" + file_name;
    createNegativeSamples(image, ojb_bbxes, negative_path);

    std::string positive_path_posi = dest_path + "/positive/" + file_name;
    std::string positive_path_part = dest_path + "/part/" + file_name;
    createPositiveSamples(image, ojb_bbxes, positive_path_posi, positive_path_part);
}

void GeneratePatch::createNegativeSamples(cv::Mat &img, std::vector<cv::Rect> &obj_bbxes, std::string file_prefix) {
    int valid_num = 0;
    int height = img.rows;
    int width = img.cols;
    //set the size boundary of bounding boxes
    int min_size = 20;
    int max_size = std::min(height/2, width/2);
    srand(time(NULL));
    while(valid_num < num_negative_) {
        //generate the size of patch randomly
        int patch_size = rand()%(max_size - min_size) + min_size;
        //randomly select the start point in image
        int x1 = rand()%(width - patch_size);
        int y1 = rand()%(height - patch_size);
        cv::Rect patch_bbx(x1, y1, patch_size, patch_size);
        float patch_iou = GEOMETRYTOOLS::regionsIOU(obj_bbxes, patch_bbx);
        if(patch_iou < neg_IOU_) {
            char name_suffix[32];
            std::sprintf(name_suffix, "_%03d.png", valid_num);
            std::string img_name = file_prefix + std::string(name_suffix);
            cv::Mat img_patch = img(patch_bbx).clone();
            cv::imwrite(img_name, img_patch);
            valid_num++;
        }
    }
}

void GeneratePatch::createPositiveSamples(cv::Mat &img, std::vector<cv::Rect> &obj_bbxes, std::string dst_posi, std::string dst_part) {
    //crop image around each bounding box
    for(int iter = 0; iter < int(obj_bbxes.size()); iter++) {
        int x1 = obj_bbxes[iter].x;
        int y1 = obj_bbxes[iter].y;
        int width = obj_bbxes[iter].width;
        int height = obj_bbxes[iter].height;
        if(x1 < 0 || y1 < 0) continue;
        if(std::max(width, height) < 20 ) continue;
        //set teh w and h
        srand(time(NULL));
        int min_size = int((width, height)*0.8);
        int max_size = int((width, height)*1.25);
        int part_num = 0;
        int posi_num = 0;
        for(int jter = 0;jter < num_positive_;jter++) {
            int patch_size = rand()%(max_size - min_size) + min_size;
            int delta_x = int(rand()%(int(width*0.4)) - 0.2*width);
            int delta_y = int(rand()%(int(height*0.4)) - 0.2*height);
            int x_l = std::max(0, x1  + width/2 - delta_x - patch_size/2);
            int y_l = std::max(0, y1 + height/2 - delta_y - patch_size/2);
            cv::Rect patch_bbx(x_l, y_l, x_l + patch_size, y_l + patch_size);
            //Decide which category the sample belong
            float patch_iou = GEOMETRYTOOLS::regionsIOU(obj_bbxes, patch_bbx);
            if(patch_iou >= pos_IOU_) {
                char name_suffix[32];
                std::sprintf(name_suffix, "_%03d.png", posi_num);
                std::string img_name = dst_posi + std::string(name_suffix);
                cv::Mat img_patch = img(patch_bbx).clone();
                cv::imwrite(img_name, img_patch);
                posi_num++;
            }
            else if(patch_iou >= part_IOU_) {
                char name_suffix[32];
                std::sprintf(name_suffix, "_%03d.png", part_num);
                std::string img_name = dst_part + std::string(name_suffix);
                cv::Mat img_patch = img(patch_bbx).clone();
                cv::imwrite(img_name, img_patch);
                part_num++;
            }
        }
    }
}
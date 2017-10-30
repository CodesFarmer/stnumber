#include "generate_patch.h"

void GeneratePatch::generate_patches(std::string filename, int img_size, std::string dst_path) {
    std::ifstream input_fid;
    input_fid.open(filename.c_str(), std::ios::in);
    std::system(std::string("mkdir -p " + dst_path + "/negative/").c_str());
    std::system(std::string("mkdir -p " + dst_path + "/positive/").c_str());
    std::system(std::string("mkdir -p " + dst_path + "/part/").c_str());
    negative_fid_.open(std::string(dst_path + "/negative.txt").c_str(), std::ios::out|std::ios::binary);
    positive_fid_.open(std::string(dst_path + "/positive.txt").c_str(), std::ios::out|std::ios::binary);
    part_fid_.open(std::string(dst_path + "/part.txt").c_str(), std::ios::out|std::ios::binary);

    std::string file_path;
    std::string img_name;
    std::string img_path;
    std::vector<cv::Rect> bounding_boxes;
    while(!input_fid.eof()) {
        input_fid>>file_path;
        input_fid>>img_name;
        bounding_boxes = get_bounding_boxes(file_path + "/anno/" + img_name + ".xml");
        img_path = file_path + "/cam0/" + img_name + ".png";

        create_patches(img_path, bounding_boxes, img_size, dst_path);
    }
    negative_fid_.close();
    positive_fid_.close();
    part_fid_.close();
}

void GeneratePatch::create_patches(std::string img_path, std::vector<cv::Rect> &ojb_bbxes, int img_size, std::string dest_path) {
    cv::Mat image = cv::imread(img_path);
    //get the details of file path
    std::string file_path;
    std::string file_name;
    std::string file_ext;
    FILEPARTS::fileparts(img_path, file_path, file_name, file_ext);

    std::string negative_path = dest_path + "/negative/" + file_name;
    create_negative_samples(image, ojb_bbxes, img_size, negative_path);

    std::string positive_path_posi = dest_path + "/positive/" + file_name;
    std::string positive_path_part = dest_path + "/part/" + file_name;
    create_positive_samples(image, ojb_bbxes, img_size, positive_path_posi, positive_path_part);
}

void GeneratePatch::create_negative_samples(cv::Mat &img, std::vector<cv::Rect> &obj_bbxes, int img_size, std::string file_prefix) {
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
            cv::resize(img_patch, img_patch, cv::Size(img_size, img_size), cv::INTER_AREA);
            cv::imwrite(img_name, img_patch);
            negative_fid_<<img_name<<" 0\n";
            valid_num++;
        }
    }
}

void GeneratePatch::create_positive_samples(cv::Mat &img, std::vector<cv::Rect> &obj_bbxes, int img_size, std::string dst_posi, std::string dst_part) {
    //crop image around each bounding box
    srand(time(NULL));
    for(int iter = 0; iter < int(obj_bbxes.size()); iter++) {
        int x1 = obj_bbxes[iter].x;
        int y1 = obj_bbxes[iter].y;
        int width = obj_bbxes[iter].width;
        int height = obj_bbxes[iter].height;
        if(x1 < 0 || y1 < 0) continue;
        if(std::max(width, height) < 20 ) continue;
        //set teh w and h
        int min_size = int(std::min(width, height)*0.8);
        int max_size = int(std::max(width, height)*1.25);
        int part_num = 0;
        int posi_num = 0;
        int jter = 0;
//        for(int jter = 0;jter < num_positive_;jter++) {
        while(jter < num_positive_) {
            int patch_size = rand()%(max_size - min_size) + min_size;
            int delta_x = int(rand()%(int(width*0.2)) - 0.1*width);
            int delta_y = int(rand()%(int(height*0.2)) - 0.1*height);
            int x_l = std::max(0, x1  + width/2 - delta_x - patch_size/2);
            int y_l = std::max(0, y1 + height/2 - delta_y - patch_size/2);
            int x_r = std::min(x_l + patch_size, img.cols - 1);//cols [0,224]
            int y_r = std::min(y_l + patch_size, img.rows - 1);
            //to keep the rigid of the object, we keep the width and height same
            int wi_p = x_r - x_l + 1;
            int hi_p = y_r - y_l + 1;
            patch_size = std::min(wi_p, hi_p);
            x_r = x_l + patch_size - 1;
            y_r = y_l + patch_size - 1;
//            cv::Rect patch_bbx(x_l, y_l, patch_size, patch_size);
            cv::Rect patch_bbx(x_l, y_l, x_r - x_l + 1, y_r - y_l + 1);
            //Decide which category the sample belong
            float patch_iou = GEOMETRYTOOLS::regionsIOU(obj_bbxes, patch_bbx);

            float offset_x1 = (obj_bbxes[iter].x - x_l)/float(patch_size);
            float offset_x2 = (obj_bbxes[iter].y - y_l)/float(patch_size);
            float offset_y1 = (obj_bbxes[iter].x + width - x_r)/float(patch_size);
            float offset_y2 = (obj_bbxes[iter].x + height - y_r)/float(patch_size);
            if(patch_iou >= pos_IOU_) {
                char name_suffix[32];
                std::sprintf(name_suffix, "_%03d.png", posi_num);
                std::string img_name = dst_posi + std::string(name_suffix);
                cv::Mat img_patch = img(patch_bbx).clone();
                cv::resize(img_patch, img_patch, cv::Size(img_size, img_size), cv::INTER_AREA);
                cv::imwrite(img_name, img_patch);
                positive_fid_<<img_name<<" 1 "<<offset_x1<<" "<<offset_x2<<" "<<offset_y1<<" "<<offset_y2<<"\n";
                posi_num++;
                jter++;
            }
            else if(patch_iou >= part_IOU_) {
                char name_suffix[32];
                std::sprintf(name_suffix, "_%03d.png", part_num);
                std::string img_name = dst_part + std::string(name_suffix);
                cv::Mat img_patch = img(patch_bbx).clone();
                cv::resize(img_patch, img_patch, cv::Size(img_size, img_size), cv::INTER_AREA);
                cv::imwrite(img_name, img_patch);
                part_fid_<<img_name<<" -1 "<<offset_x1<<" "<<offset_x2<<" "<<offset_y1<<" "<<offset_y2<<"\n";
                part_num++;
                jter++;
            }
        }
    }
}
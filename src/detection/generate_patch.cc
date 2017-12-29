#include "generate_patch.h"
//#define USE_DEPTH

void GeneratePatch::generate_patches_crop(std::string filename, int img_size, std::string dst_path, SaveMode save_mode) {
    int length = FILEPARTS::counting_lines(filename);
    std::ifstream input_fid;
    input_fid.open(filename.c_str(), std::ios::in);
    create_destination(dst_path, img_size, save_mode);

    std::string file_path;
    std::string img_name;
    std::string img_path;
    std::vector<cv::Rect> bounding_boxes;
    int cur_iter = 0;
    while(!input_fid.eof()) {
        input_fid>>file_path;
        input_fid>>img_name;
        bounding_boxes = get_bounding_boxes(file_path + "/xml/" + img_name + ".xml");
        //if there exist two or more bounding boxes, we think it is a morbid bounding boxes
//        if(bounding_boxes.size() > 1) continue;
        //if multi bbxes, we think the last one is the ground truth
        while(bounding_boxes.size() > 1) {
            bounding_boxes.erase(bounding_boxes.begin());
        }
        img_path = file_path + "/cam0/" + img_name + ".png";
        if(access(img_path.c_str(), F_OK) == -1) {
            img_path = file_path + "/cam0/" + img_name + ".jpg";
        }

        if(bounding_boxes.size() > 0) {
            create_patches(img_path, bounding_boxes, img_size, dst_path);
        }
        std::cout << "\r" << std::setprecision(4) << 100 * float(cur_iter) / float(length) << "% completed..."
                  << std::flush;
        cur_iter++;
    }
    std::cout<<std::endl;
    if(save_mode_ == DISK) {
        negative_fid_.close();
        positive_fid_.close();
        part_fid_.close();
    }
    else if(save_mode_ == HDF5) {
        hdf5_writer_->close_hdf5();
    }
}
/*
 * Generate patches according to the ground truth of image
 * including negative samples
 * including positive and partial samples
 * */
void GeneratePatch::create_patches(std::string img_path, std::vector<cv::Rect> &ojb_bbxes, int img_size, std::string dest_path) {
    //Read the infrared image
    //check if the image is exist
    if(access(img_path.c_str(), F_OK) == -1) return;
    cv::Mat infrared = cv::imread(img_path, CV_8UC1);
    if(infrared.empty()) return;
    cv::Mat image = infrared.clone();
#ifdef USE_DEPTH
    //Read the depth image
    std::string img_depth_path(img_path);
    FILEPARTS::replace_string(img_depth_path, "cam0", "dep0");
    FILEPARTS::replace_string(img_depth_path, "jpg", "png");
    //check if the image is exist
    if(access(img_depth_path.c_str(), F_OK) == -1) return;
    cv::Mat depth = cv::imread(img_depth_path, CV_16UC1);
    if(depth.empty()) return;
    cv::bitwise_and(depth, 0x1FFF, depth);
    //Normalize the data into float32 and merge depth and irfrared into one image
    merge_image(infrared, depth, image);
#endif //USE_DEPTH

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
    srand(time(NULL));
    float low_bound = 500.0f;
    //statistic the sum of rows and columns
    int pt_l = 0;
    int pt_r = img.rows;
    int pt_t = 0;
    int pt_b = img.cols;
    get_heat_region(img, low_bound, pt_l, pt_t, pt_r, pt_b);
    //rows l r height
    //cols t b width

    while(valid_num < num_negative_) {
        //set the size boundary of bounding boxes
        int min_size = 20;
        int max_size = std::min(height/2, width/2);
        //generate the size of patch randomly
        int patch_size = rand()%(max_size - min_size) + min_size;
        //randomly select the start point in image
        int x1 = rand()%(width - patch_size);
        int y1 = rand()%(height - patch_size);
        //Here we generate the start point near the region of high light region
        if(rand()%3 == 0) {
            max_size = std::min((pt_r - pt_l + 1), (pt_b - pt_t + 1));
            if(max_size - min_size > img_size) {
                //generate the size of patch randomly
                patch_size = rand()%(max_size - min_size) + min_size;
                //Generate starting point
                int rh = pt_r - pt_l + 1;
                int rw = pt_b - pt_t + 1;
                //randomly select the start point in image
                int height_reduce = 0;
                if(height - pt_r <= patch_size) {
                    height_reduce = patch_size;
                }
                int width_reduce = 0;
                if(width - pt_b <= patch_size) {
                    width_reduce = patch_size;
                }
                x1 = rand() % (rw - width_reduce) + pt_t;
                y1 = rand() % (rh - height_reduce) + pt_l;
            }
        }
        cv::Rect patch_bbx(x1, y1, patch_size, patch_size);
        int index = 0;
        float patch_iou = GEOMETRYTOOLS::regionsIOU(obj_bbxes, patch_bbx, index);
        if(patch_iou < neg_IOU_) {
            char name_suffix[32];
            std::sprintf(name_suffix, "_%03d", valid_num);
            std::string img_name = file_prefix + std::string(name_suffix);
            write_to_disk(img, img_size, img_name, 0, true, patch_bbx, patch_bbx);
            valid_num++;
        }
    }
}

/*
 * The function create positive or part samples according to the original image and the ground truth bounding box
 * --img            the original image
 * --obj_bbxes      bounding box of ground truth
 * --img_size       destination images with fixed size
 * --dest_posi      the path to write the positive images
 * --dst_part       the path to write the partial images
 */
void GeneratePatch::create_positive_samples(cv::Mat &img, std::vector<cv::Rect> &obj_bbxes, int img_size,
                                            std::string dst_posi, std::string dst_part) {
    float max_shift = 0.1;
    //crop image around each bounding box
    srand(time(NULL));
    for(int iter = 0; iter < int(obj_bbxes.size()); iter++) {
        int x1 = obj_bbxes[iter].x;
        int y1 = obj_bbxes[iter].y;
        int width = obj_bbxes[iter].width;
        int height = obj_bbxes[iter].height;
        if(obj_bbxes[iter].width <= 5 || obj_bbxes[iter].height <= 5) continue;
        if(x1 < 0 || y1 < 0) continue;
        if(std::max(width, height) < 20 ) continue;
        //set teh w and h
        int min_size = int(std::min(width, height)*0.8);
        int max_size = int(std::max(width, height)*1.25);
        int part_num = 0;
        int posi_num = 0;
        int jter = 0;
//        for(int jter = 0;jter < num_positive_;jter++) {
        struct timeval formertime;
        struct timeval curtime;
        gettimeofday(&formertime, NULL);
        while(jter < num_positive_) {
            gettimeofday(&curtime, NULL);
            double time_cost = (curtime.tv_sec - formertime.tv_sec) + (curtime.tv_usec - formertime.tv_usec)/1000000.0;
            if(time_cost > 1.0) break;
            int patch_size = rand()%(max_size - min_size) + min_size;
            int delta_x = int(rand()%(int(width*max_shift*2.0)) - max_shift*width);
            int delta_y = int(rand()%(int(height*max_shift*2.0)) - max_shift*height);
            int x_l = std::max(0, x1  + width/2 - delta_x - patch_size/2);
            int y_l = std::max(0, y1 + height/2 - delta_y - patch_size/2);
            int x_r = std::min(x_l + patch_size, img.cols - 1);//cols [0,224]
            int y_r = std::min(y_l + patch_size, img.rows - 1);
            //to keep the rigid of the object, we keep the width and height same
            int wi_p = x_r - x_l + 1;
            int hi_p = y_r - y_l + 1;
            patch_size = std::max(wi_p, hi_p);

            if(wi_p > hi_p) {
                y_l = y_l - (wi_p - hi_p)/2;
            }
            if(hi_p > wi_p) {
                x_l = x_l - (hi_p - wi_p)/2;
            }
            x_r = x_l + patch_size;
            y_r = y_l + patch_size;

            if(x_r >= img.cols) {
                x_l = img.cols - patch_size;
                x_r = img.cols - 1;
            }
            else {
                ;
            }
            if(y_r >= img.rows) {
                y_l = img.rows - patch_size;
                y_r = img.rows - 1;
            }
//            cv::Rect patch_bbx(x_l, y_l, patch_size, patch_size);
            cv::Rect patch_bbx(x_l, y_l, x_r - x_l + 1, y_r - y_l + 1);
            //Decide which category the sample belong
            int index = 0;
            float patch_iou = GEOMETRYTOOLS::regionsIOU(obj_bbxes, patch_bbx, index);

            if(patch_iou >= pos_IOU_) {
                char name_suffix[32];
                std::sprintf(name_suffix, "_%03d", posi_num);
                std::string img_name = dst_posi + std::string(name_suffix);
                write_to_disk(img, img_size, img_name, 1, true, cv::Rect(x_l, y_l, x_r - x_l, y_r - y_l), obj_bbxes[iter]);
                posi_num++;
                jter++;
            }
            else if(patch_iou >= part_IOU_) {
                char name_suffix[32];
                std::sprintf(name_suffix, "_%03d", part_num);
                std::string img_name = dst_part + std::string(name_suffix);
                write_to_disk(img, img_size, img_name, -1, true, cv::Rect(x_l, y_l, x_r - x_l, y_r - y_l), obj_bbxes[iter]);
                part_num++;
                jter++;
            }
        }
    }
}

void GeneratePatch::initialize_detector(const std::map<std::string, std::pair<std::string, std::string> > & model_path,
                                        const float img2net_scale,
                                        const std::vector<float> mean_value) {
    int channels = 1;
#ifdef USE_DEPTH
    channels = 2;
#endif
    detector_ = boost::make_shared<FaceDetector<float> >(channels);
    detector_->initialize_network(model_path);
    detector_->initialize_transformer(img2net_scale, mean_value);
}

/*
 * Generate patches with MTCNN
 * The function forward an image through mtcnn with and get the patches generated from mtcnn
 */
void GeneratePatch::generate_patches_cnn(std::string filename, int img_size, std::string dst_path, SaveMode save_mode) {
    int length = FILEPARTS::counting_lines(filename);
    std::ifstream input_fid;
    input_fid.open(filename.c_str(), std::ios::in);

    create_destination(dst_path, img_size, save_mode);

    std::string file_path;
    std::string img_name;
    std::string img_path;
    std::vector<cv::Rect> bounding_boxes;
    int cur_iter = 0;
    while(!input_fid.eof()) {
        input_fid>>file_path;
        input_fid>>img_name;
        bounding_boxes = get_bounding_boxes(file_path + "/xml/" + img_name + ".xml");
        while(bounding_boxes.size() > 1) {
            bounding_boxes.erase(bounding_boxes.begin());
        }

        img_path = file_path + "/cam0/" + img_name + ".png";

        if(access(img_path.c_str(), F_OK) == -1) {
            img_path = file_path + "/cam0/" + img_name + ".jpg";
        }
//        img_path = "/home/slam/DepthGesture/stnumber/build/1509071930041880334.png";
//        bounding_boxes = get_bounding_boxes("/home/slam/DepthGesture/stnumber/build/1509071930041880334.xml");
        //Read the infrared image at first
        cv::Mat image_ir = cv::imread(img_path, CV_8UC1);
        if(image_ir.empty()) continue;
        //Before input into neural network
        float mean_ir = 0.0f;
        float scale_ir = 0.0125f;
        cv::Mat tmpimg_ir = image_ir.clone();
        cv::Mat img_ir_float(tmpimg_ir.size(), CV_32FC1);
        tmpimg_ir.convertTo(img_ir_float, CV_32FC1);
        img_ir_float = (img_ir_float - mean_ir)*scale_ir;

        cv::Mat image = image_ir.clone();
        cv::Mat image_float = img_ir_float.clone();
#ifdef USE_DEPTH
        //Read the depth image
        FILEPARTS::replace_string(img_path, "cam0", "dep0");
        FILEPARTS::replace_string(img_path, "jpg", "png");
        cv::Mat image_dp = cv::imread(img_path, CV_16UC1);
        if(image_dp.empty()) continue;
        cv::bitwise_and(image_dp, 0x1FFF, image_dp);
        //Before input into neural network
        float mean_dp = 0.0f;
        float scale_dp = 0.00083f;
        cv::Mat tmpimg_dp = image_dp.clone();
        cv::Mat img_dp_float(tmpimg_dp.size(), CV_32FC1);
        tmpimg_dp.convertTo(img_dp_float, CV_32FC1);
        img_dp_float = (img_dp_float - mean_dp)*scale_dp;
        merge_image(img_ir_float, img_dp_float, image_float);
        merge_image(image_ir, image_dp, image);
#endif //USE_DEPTH

        //Forward pass the neural network
        std::vector<std::vector<float> > hand_bbx =  detector_->detect_face(image_float);
        //get the details of file path
        std::string file_path;
        std::string file_name;
        std::string file_ext;
        FILEPARTS::fileparts(img_path, file_path, file_name, file_ext);
        std::string negative_path = dst_path + "/negative/" + file_name;
        std::string positive_path_posi = dst_path + "/positive/" + file_name;
        std::string positive_path_part = dst_path + "/part/" + file_name;

//        std::printf("The size of bbx is %d", int(hand_bbx.size()));
        //classifiy the bounding boxes according to the iou of bounding boxes
        for(int bbx_id = 0; bbx_id < hand_bbx.size(); bbx_id ++) {
            int x_l = std::max( hand_bbx[bbx_id][0], 0.0f );
            int y_l = std::max( hand_bbx[bbx_id][1], 0.0f);
            int x_r = std::min( hand_bbx[bbx_id][2], float(image.cols-1) );
            int y_r = std::min( hand_bbx[bbx_id][3], float(image.rows-1) );
            cv::Rect hand_rect(x_l, y_l, x_r - x_l + 1, y_r - y_l + 1);
//            cv::rectangle(image, hand_rect, cv::Scalar(255));
//            cv::imshow("test", image);
//            cv::waitKey(0);
            int index;
            float patch_iou = GEOMETRYTOOLS::regionsIOU(bounding_boxes, hand_rect, index);

            //if iou is under neg_IOU_, then this is a negative sample
            if(patch_iou < neg_IOU_) {
                char name_suffix[32];
                std::sprintf(name_suffix, "_%03d", bbx_id);
                std::string img_name = negative_path + std::string(name_suffix);
                write_to_disk(image, img_size, img_name, 0, true, hand_rect, hand_rect);
            }
            else if(patch_iou >= part_IOU_) {
                //if it is a positive sample
                if(patch_iou >= pos_IOU_) {
                    char name_suffix[32];
                    std::sprintf(name_suffix, "_%03d", bbx_id);
                    std::string img_name = positive_path_posi + std::string(name_suffix);
                    write_to_disk(image, img_size, img_name, 1, true, hand_rect, bounding_boxes[index]);
                }//if it is a part appearance sample
                else if(patch_iou >= part_IOU_) {
                    char name_suffix[32];
                    std::sprintf(name_suffix, "_%03d", bbx_id);
                    std::string img_name = positive_path_part + std::string(name_suffix);
                    write_to_disk(image, img_size, img_name, 1, true, hand_rect, bounding_boxes[index]);
                }
            }
        }

        if(bounding_boxes.size() > 0 && hand_bbx.size() == 0) {
            create_patches(img_path, bounding_boxes, img_size, dst_path);
        }
        std::cout << "\r" << std::setprecision(4) << 100 * float(cur_iter) / float(length) << "% completed..."
                  << std::flush;
        cur_iter++;
    }
    std::cout<<std::endl;
    if(save_mode_ == DISK) {
        negative_fid_.close();
        positive_fid_.close();
        part_fid_.close();
    }
    else if(save_mode_ == HDF5) {
        hdf5_writer_->close_hdf5();
    }
}

/*
 * This function write the data sets at four direction into disk
 * input:
 * --image          original image
 * --rect_sample    the sample rectangle
 * --rect_ground    the ground truth bounding box
 * --label          label the the sample (negative, part, positive)
 * --augmentation   whether to augment the data set into 4 direction or not
 * --img_size       finally size of the samples
 */
void GeneratePatch::write_to_disk(const cv::Mat & image, int img_size, const std::string & name_prefix, int label,
                                  const bool augmentation, const cv::Rect &rect_sample, const cv::Rect &rect_ground) {
    float bias = 1e-30;
    //Write the original image
    int xs_l = rect_sample.x;
    int ys_l = rect_sample.y;
    int xs_r = rect_sample.x + rect_sample.width;
    int ys_r = rect_sample.y + rect_sample.height;
    int xg_l = rect_ground.x;
    int yg_l = rect_ground.y;
    int xg_r = rect_ground.x + rect_ground.width;
    int yg_r = rect_ground.y + rect_ground.height;
    int aug_id = 0;
    char name_suffix[32];
    std::sprintf(name_suffix, "_%02d.png", aug_id);
    std::string img_name = name_prefix + std::string(name_suffix);
    cv::Mat img_patch = image(rect_sample).clone();
//    cv::imshow("test", img_patch);
//    cv::waitKey(0);
    cv::resize(img_patch, img_patch, cv::Size(img_size, img_size), cv::INTER_AREA);
//    cv::imwrite(img_name, img_patch);
    float offset_x1 = (xg_l - xs_l) / (xs_r - xs_l + bias);
    float offset_y1 = (yg_l - ys_l) / (ys_r - ys_l + bias);
    float offset_x2 = (xg_r - xs_r) / (xs_r - xs_l + bias);
    float offset_y2 = (yg_r - ys_r) / (ys_r - ys_l + bias);
    std::vector<float> labels(5, 0);
    labels[0] = label;
    labels[1] = offset_x1;
    labels[2] = offset_y1;
    labels[3] = offset_x2;
    labels[4] = offset_y2;
    if(save_mode_ == DISK) {
        save2disk(img_patch, labels, img_name);
    }
    else if(save_mode_ == HDF5) {
        save2hdf5(img_patch, labels);
    }

    //Augment the data into 4 directions
    if(augmentation) {
        //image transpose
        cv::Mat img_10;
        img_10 = img_patch.clone();
        cv::transpose(img_10, img_10);
        int x_l1 = ys_l;
        int y_l1 = xs_l;
        int x_r1 = ys_r;
        int y_r1 = xs_r;
        offset_x1 = (yg_l - x_l1) / (x_r1 - x_l1 + bias);
        offset_y1 = (xg_l - y_l1) / (y_r1 - y_l1 + bias);
        offset_x2 = (yg_r - x_r1) / (x_r1 - x_l1 + bias);
        offset_y2 = (xg_r - y_r1) / (y_r1 - y_l1 + bias);
        aug_id++;
        std::sprintf(name_suffix, "_%02d.png", aug_id);
        img_name = name_prefix + std::string(name_suffix);
//        cv::imwrite(img_name, img_10);
        labels[0] = label;
        labels[1] = offset_x1;
        labels[2] = offset_y1;
        labels[3] = offset_x2;
        labels[4] = offset_y2;
        if(save_mode_ == DISK) {
            save2disk(img_10, labels, img_name);
        }
        else if(save_mode_ == HDF5) {
            save2hdf5(img_10, labels);
        }

        //image flip vertically
        cv::Mat img_01;
        img_01 = img_patch.clone();
        cv::flip(img_01, img_01, 0);
        int x_l2 = xs_l;
        int y_l2 = image.rows - ys_r;
        int x_r2 = xs_r;
        int y_r2 = image.rows - ys_l;
        offset_x1 = (xg_l - x_l2) / (x_r2 - x_l2 + bias);
        offset_y1 = (image.rows - yg_r - y_l2) / (y_r2 - y_l2 + bias);
        offset_x2 = (xg_r - x_l2) / (x_r2 - x_l2 + bias);
        offset_y2 = (image.rows - yg_l - y_l2) / (y_r2 - y_l2 + bias);
        aug_id++;
        std::sprintf(name_suffix, "_%02d.png", aug_id);
        img_name = name_prefix + std::string(name_suffix);
//        cv::imwrite(img_name, img_01);
        labels[0] = label;
        labels[1] = offset_x1;
        labels[2] = offset_y1;
        labels[3] = offset_x2;
        labels[4] = offset_y2;
        if(save_mode_ == DISK) {
            save2disk(img_01, labels, img_name);
        }
        else if(save_mode_ == HDF5) {
            save2hdf5(img_01, labels);
        }

        //image transpose and flip vertically
        cv::Mat img_11;
        img_11 = img_patch.clone();
        cv::transpose(img_11, img_11);
        cv::flip(img_11, img_11, 1);
        int x_l3 = image.rows - ys_r;
        int y_l3 = xs_l;
        int x_r3 = image.rows - ys_l;
        int y_r3 = xs_r;
        offset_x1 = (image.rows - yg_r - x_l3) / float(x_r3 - x_l3 + bias);
        offset_y1 = (xg_l - y_l3) / float(y_r3 - y_l3 + bias);
        offset_x2 = (image.rows - yg_l - x_r3) / float(x_r3 - x_l3 + bias);
        offset_y2 = (xg_r - y_r3) / float(y_r3 - y_l3);
        aug_id++;
        std::sprintf(name_suffix, "_%02d.png", aug_id);
        img_name = name_prefix + std::string(name_suffix);
        labels[0] = label;
        labels[1] = offset_x1;
        labels[2] = offset_y1;
        labels[3] = offset_x2;
        labels[4] = offset_y2;
        if(save_mode_ == DISK) {
            save2disk(img_11, labels, img_name);
        }
        else if(save_mode_ == HDF5) {
            save2hdf5(img_11, labels);
        }
    }
}

//bool GeneratePatch::hdf5_writer_16u28u(cv::Mat &img_16u, cv::Mat &img_8u) {
//    cv::Mat img_src = img_16u.clone();
//    cv::bitwise_and(img_16u, 0x1FFF, img_src);
//    img_src.convertTo(img_8u, CV_32FC1, 0.25);
//}

void GeneratePatch::save2hdf5(const cv::Mat &image, const std::vector<float> label) {
    cv::Mat tempimage = image.clone();
    image.convertTo(tempimage, CV_32FC1);
    cv::Mat normalized_img = tempimage.clone();
#ifdef INPUT_L2NORM
    cv::normalize(tempimage, normalized_img);
#endif
    hdf5_writer_->write_tuples(normalized_img, label);
}

void GeneratePatch::save2disk(const cv::Mat &image, const std::vector<float> labels, const std::string &img_name) {
    cv::Mat img;
    image.convertTo(img, CV_8UC1);
    cv::imwrite(img_name, img);
    int label = int(labels[0]);
    if(label == 0) negative_fid_<< img_name <<"\n";
    else if(label == -1) part_fid_<< img_name <<" "<<label<<" "<<labels[0]<<" "<<labels[1]<<" "<<labels[2]<<" "<<labels[3]<<"\n";
    else if(label == 1) positive_fid_<< img_name <<" "<<label<<" "<<labels[0]<<" "<<labels[1]<<" "<<labels[2]<<" "<<labels[3]<<"\n";
}

//void GeneratePatch::merge_image(const cv::Mat & img_8u, const cv::Mat & img_16u, cv::Mat &image) {
void GeneratePatch::merge_image(cv::Mat & img_8u, cv::Mat & img_16u, cv::Mat &image) {
    //Normalize the data into float32
    img_8u.convertTo(img_8u, CV_32FC1);
    img_16u.convertTo(img_16u, CV_32FC1);
    std::vector<cv::Mat> ir_dep;
    ir_dep.push_back(img_8u);
    ir_dep.push_back(img_16u);
//    cv::Mat image(img_8u.rows, img_8u.cols, CV_32FC2);
    image.convertTo(image, CV_32FC2);
    cv::merge(ir_dep, image);
}

void GeneratePatch::create_destination(const std::string &dst_path, int img_size, SaveMode save_mode) {
    if(save_mode != DISK && save_mode != HDF5) {
        throw std::invalid_argument("Wrong save mode(GeneratePatch::DISK or GeneratePatch::HDF5)");
    }
    save_mode_ = save_mode;
    //if the data will be written into disk, create the input handle
    if(save_mode_ == DISK) {

        std::system(std::string("mkdir -p " + dst_path + "/negative/").c_str());
        std::system(std::string("mkdir -p " + dst_path + "/positive/").c_str());
        std::system(std::string("mkdir -p " + dst_path + "/part/").c_str());

        //create file lists
        if(access(std::string(dst_path + "/negative.txt").c_str(), F_OK) == -1){
            negative_fid_.open(std::string(dst_path + "/negative.txt").c_str(), std::ios::out|std::ios::binary);
        }
        else {
            negative_fid_.open(std::string(dst_path + "/negative.txt").c_str(), std::ios::out|std::ios::binary|std::ios::app);
        }
        if(access(std::string(dst_path + "/positive.txt").c_str(), F_OK) == -1){
            positive_fid_.open(std::string(dst_path + "/positive.txt").c_str(), std::ios::out|std::ios::binary);
        }
        else {
            positive_fid_.open(std::string(dst_path + "/positive.txt").c_str(), std::ios::out|std::ios::binary|std::ios::app);
        }
        if(access(std::string(dst_path + "/part.txt").c_str(), F_OK) == -1){
            part_fid_.open(std::string(dst_path + "/part.txt").c_str(), std::ios::out|std::ios::binary);
        }
        else {
            part_fid_.open(std::string(dst_path + "/part.txt").c_str(), std::ios::out|std::ios::binary|std::ios::app);
        }
    }
    else if(save_mode_ == HDF5) {
        //create HDF5 file
        //Setup the dimension of input data
        std::vector<int> data_dimension;
        data_dimension.push_back(1);//num of batches
        data_dimension.push_back(1);//channels
#ifdef USE_DEPTH
        data_dimension[data_dimension.size()-1] = 2;//channels
#endif //USE_DEPTH
        data_dimension.push_back(img_size);//height
        data_dimension.push_back(img_size);//width
        //Setup the size of label
        std::vector<int> label_dimension;
        label_dimension.push_back(1);
        label_dimension.push_back(5);
        //create the data sets
        std::vector<float> mean_value;
        std::vector<float> shrink_ratio;
//    mean_value.push_back(23.9459f);
//    mean_value.push_back(474.2429f);
//    shrink_ratio.push_back(0.0125f);
//    shrink_ratio.push_back(0.00083f);
        mean_value.push_back(0.0f);
        mean_value.push_back(0.0f);
        shrink_ratio.push_back(0.0125f);
        shrink_ratio.push_back(0.00083f);
#ifdef INPUT_L2NORM
        shrink_ratio[0] = 1.0f;
        shrink_ratio[1] = 1.0f;
        std::printf("Input image are l2 normalized...\n");
#endif
        hdf5_writer_ = boost::make_shared<Mat2H5>(mean_value, shrink_ratio, 1000);
        hdf5_writer_->create_hdf5(dst_path);
        hdf5_writer_->create_dataset(Mat2H5::DATA, data_dimension, "float");
        hdf5_writer_->create_dataset(Mat2H5::LABEL, label_dimension, "float");
        save_mode_ = HDF5;
    }
}


void GeneratePatch::get_heat_region(cv::Mat & img, float low_bound, int &pt_l, int &pt_t, int &pt_r, int & pt_b) {
    //statistic the sum of rows and columns
    cv::Mat rows_sum(cv::Size(img.rows, 1), CV_32FC1);
    cv::Mat column_sum(cv::Size(1, img.cols), CV_32FC1);
    cv::Mat hist_img;
    img.convertTo(hist_img, CV_32FC1);
    cv::reduce(hist_img, rows_sum, 0, CV_REDUCE_SUM);
    cv::reduce(hist_img, column_sum, 1, CV_REDUCE_SUM);
    pt_l = 0;
    pt_r = img.rows;
    pt_t = 0;
    pt_b = img.cols;
    int start_pt = 0;
    while(start_pt < img.rows - 1 && img.at<float>(start_pt, 0) < low_bound) start_pt++;
    pt_l = start_pt;
    start_pt = img.rows - 1;
    while(start_pt > 0 && img.at<float>(start_pt, 0) < low_bound) start_pt--;
    pt_r = start_pt;
    start_pt = 0;
    while(start_pt < img.cols - 1 && img.at<float>(0, start_pt) < low_bound) start_pt++;
    pt_t = start_pt;
    start_pt = img.cols - 1;
    while(start_pt >0 && img.at<float>(0, start_pt) < low_bound) start_pt--;
    pt_b = start_pt;
}

/*
 * The function generate samples from images with ground truth and bounding boxes stored in text
 * In text, there are 5 component of a line
 * image_path x1 y1 x2 y2
 * The path of xml file can be obtained by replace the word 'png|jpg|cam0' in image_path with xml
 * */
void GeneratePatch::generate_patches_text(std::string filename, int img_size, std::string dst_path, SaveMode save_mode) {
    int length = FILEPARTS::counting_lines(filename);
    std::ifstream input_fid;
    input_fid.open(filename.c_str(), std::ios::in);

    create_destination(dst_path, img_size, save_mode);
    std::string img_name;
    std::string img_path;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    std::vector<cv::Rect> bounding_boxes;
    int cur_iter = 0;
    while(!input_fid.eof()) {
        input_fid>>img_path;
        input_fid>>x1;
        input_fid>>y1;
        input_fid>>x2;
        input_fid>>y2;
        std::string xml_path;
        xml_path = img_path;
        FILEPARTS::replace_string(xml_path, "cam0", "xml");
        FILEPARTS::replace_string(xml_path, "png", "xml");
        FILEPARTS::replace_string(xml_path, "jpg", "xml");
        bounding_boxes = get_bounding_boxes(xml_path);
        cv::Mat image = cv::imread(img_path, CV_8UC1);
        x1 = std::max(x1, 0.0f);
        y1 = std::max(y1, 0.0f);
        x2 = std::min(x2, image.cols-1.0f);
        y2 = std::min(y2, image.rows-1.0f);
        cv::Rect hand_rect(x1, y1, x2 - x1, y2 - y1);
        cv::Mat image_rgb;
        cv::cvtColor(image, image_rgb, cv::COLOR_GRAY2BGR);
        cv::rectangle(image_rgb, hand_rect, cv::Scalar(0, 0, 255));
        cv::rectangle(image_rgb, bounding_boxes[0], cv::Scalar(0, 255, 0));
        cv::imshow("bbx", image_rgb);
        cv::waitKey(1);

        std::string file_path;
        std::string file_name;
        std::string file_ext;
        FILEPARTS::fileparts(img_path, file_path, file_name, file_ext);
        std::string negative_path = dst_path + "/negative/" + file_name;
        std::string positive_path_posi = dst_path + "/positive/" + file_name;
        std::string positive_path_part = dst_path + "/part/" + file_name;
        int index;
        float patch_iou = GEOMETRYTOOLS::regionsIOU(bounding_boxes, hand_rect, index);

        //if iou is under neg_IOU_, then this is a negative sample
        if(patch_iou < neg_IOU_) {
            char name_suffix[32];
            std::sprintf(name_suffix, "_%03d", 0);
            std::string img_name = negative_path + std::string(name_suffix);
            write_to_disk(image, img_size, img_name, 0, true, hand_rect, hand_rect);
        }
        else if(patch_iou >= part_IOU_) {
            //if it is a positive sample
            if(patch_iou >= pos_IOU_) {
                char name_suffix[32];
                std::sprintf(name_suffix, "_%03d", 0);
                std::string img_name = positive_path_posi + std::string(name_suffix);
                write_to_disk(image, img_size, img_name, 1, true, hand_rect, bounding_boxes[index]);
            }//if it is a part appearance sample
            else if(patch_iou >= part_IOU_) {
                char name_suffix[32];
                std::sprintf(name_suffix, "_%03d", 0);
                std::string img_name = positive_path_part + std::string(name_suffix);
                write_to_disk(image, img_size, img_name, 1, true, hand_rect, bounding_boxes[index]);
            }
        }

        std::cout << "\r" << std::setprecision(4) << 100 * float(cur_iter) / float(length) << "% completed..."
                  << std::flush;
        cur_iter++;
    }
}
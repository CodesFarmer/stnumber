//The class FaceDetector employ the MTCNN network to detect the faces in images
//We utilize the pre-trained model for detecting faces
//The MTCNN is made up by 3 networks, specifically, PNet, RNet and ONet.
//PNet is proposal network which will generate fixed number of bounding boxes
//RNet is refine network which will refine the generated bounding boxes
//ONet is the output network, which will further filter out some bounding boxes
#pragma(once)
#include <vector>
#include <map>
#include <stdexcept>
#include <boost/algorithm/string.hpp>

#include <opencv2/opencv.hpp>
#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/memory_data_layer.hpp"

#include "detect_geometry.h"

template<typename Dtype> class FaceDetector {
public:
	struct Bbox{
		int x1;
		int y1;
		int x2;
		int y2;
		Bbox(int xs, int ys, int xe, int ye):x1(xs), y1(ys), x2(xe), y2(ye){};
	};
public:
    FaceDetector(int channels=1):input_channels_(channels), has_tnet_(true){}
	int initialize_network(std::map<std::string, std::pair<std::string, std::string> > netpath) {
		//First of all, we will check if the model and model proto exist
		std::map<std::string, std::pair<std::string, std::string> >::iterator iter;
        if(netpath.count("tnet") == 0) has_tnet_ = false;
		for(iter = netpath.begin();iter!=netpath.end();iter++) {
			if(access(iter->second.first.c_str(), F_OK) == -1) {
				if(!boost::iequals(iter->first, "tnet")) {
					std::cerr << "The model " << iter->second.first << " does not exist..." << std::endl;
					throw std::invalid_argument("");
//					return -1;
				}
				else {
					has_tnet_ = false;
					std::cerr << "WARING: The model " << iter->second.first << " does not exist... ";
					std::cerr << "We will use ONet for tracking, while it will slow down the program..." << std::endl;
					getchar();
				}
			}
			if(access(iter->second.second.c_str(), F_OK) == -1) {
				if(!boost::iequals(iter->first, "tnet")) {
					std::cerr << "The model proto " << iter->second.second << " does not exist..." << std::endl;
					throw std::invalid_argument("");
//						return -1;
				}
				else {
					has_tnet_ = false;
					std::cerr << "WARING: The model proto " << iter->second.second << " does not exist... ";
					std::cerr << "We will use ONet for tracking, while it will slow down the program..." << std::endl;
				}
			}
		}
		//We will initialize each network respectively
		//PNet
		try{
			boost::shared_ptr<caffe::Net<Dtype> > PNet(new caffe::Net<Dtype>(netpath["pnet"].second, caffe::TEST, 0, NULL));
			// boost::shared_ptr<caffe::Net<Dtype> > PNet(new caffe::Net<Dtype>(netpath["pnet"].second, caffe::TEST, 0, NULL, NULL));
			PNet_ = PNet;
			PNet_->CopyTrainedLayersFrom(netpath["pnet"].first);
		}
		catch(int error_){
			std::cerr<<"Fatal error wihile initializing network PNet..."<<std::endl;
			return -1;
		}
		//RNet
		try{
			boost::shared_ptr<caffe::Net<Dtype> > RNet(new caffe::Net<Dtype>(netpath["rnet"].second, caffe::TEST, 0, NULL));
			// boost::shared_ptr<caffe::Net<Dtype> > RNet(new caffe::Net<Dtype>(netpath["rnet"].second, caffe::TEST, 0, NULL, NULL));
			RNet_ = RNet;
			RNet_->CopyTrainedLayersFrom(netpath["rnet"].first);
		}
		catch(int error_){
			std::cerr<<"Fatal error wihile initializing network RNet..."<<std::endl;
			return -1;
		}
		//ONet
		try{
			boost::shared_ptr<caffe::Net<Dtype> > ONet(new caffe::Net<Dtype>(netpath["onet"].second, caffe::TEST, 0, NULL));
			// boost::shared_ptr<caffe::Net<Dtype> > ONet(new caffe::Net<Dtype>(netpath["onet"].second, caffe::TEST, 0, NULL, NULL));
			ONet_ = ONet;
			ONet_->CopyTrainedLayersFrom(netpath["onet"].first);
		}
		catch(int error_){
			std::cerr<<"Fatal error wihile initializing network ONet..."<<std::endl;
			return -1;
		}
		//TNet
        if(has_tnet_) {
            try {
                boost::shared_ptr<caffe::Net<Dtype> > TNet(
                        new caffe::Net<Dtype>(netpath["tnet"].second, caffe::TEST, 0, NULL));
                // boost::shared_ptr<caffe::Net<Dtype> > TNet(new caffe::Net<Dtype>(netpath["onet"].second, caffe::TEST, 0, NULL, NULL));
                TNet_ = TNet;
                TNet_->CopyTrainedLayersFrom(netpath["tnet"].first);
            }
            catch (int error_) {
                std::cerr << "Fatal error wihile initializing network ONet..." << std::endl;
                return -1;
            }
        }
		return 0;
	};
	//This function initialize a transformer which will transform the data to meet the require of network
	void initialize_transformer(const float img2net_scale, const std::vector<float> mean_value) {
		caffe::TransformationParameter transform_params;
		transform_params.set_scale(img2net_scale);
		for(size_t iter = 0;iter<mean_value.size();iter++){
			transform_params.add_mean_value(mean_value[iter]);
		}
		boost::shared_ptr<caffe::DataTransformer<Dtype> > tmp_transform(new caffe::DataTransformer<Dtype>(transform_params, caffe::TEST));
		transformer_ = tmp_transform;
	};
	std::vector<std::vector<Dtype> > tracking_hand(const cv::Mat& image, const std::vector<float> & rect, float threshold = 0.6) {
		//declare the variable to store the bounding box with probability
		std::vector<std::vector<Dtype> > all_boxes;
        cv::Mat img(image.rows, image.cols, image.depth());
        image.copyTo(img);
        cv::transpose(img, img);
		//If the likehood of the rect belong to a hand under the threshold, we refind it with MTCNN
		if(rect[4] < threshold) {
			std::printf("MTCNN\n");
            cv::transpose(img, img);
			all_boxes = detect_face(img);
			return all_boxes;
		}
//		//Transfer the rect into std::vector
        all_boxes.push_back(rect);
		//If there exist tnet, we tracking object with tnet, else we use onet
//		has_tnet_ = false;
		if(has_tnet_) {
			all_boxes = tracking_bboxes(img, all_boxes, 0.0);
		}
		else {
			std::printf("ONET\n");
//            double t = (double)cvGetTickCount();
			all_boxes = output_bboxes(img, all_boxes, 0.0);
//            t = (double)cvGetTickCount() - t;
//            printf( "run time = %gms\n", t/(cvGetTickFrequency()*1000));
		}
		return all_boxes;
	}
	std::vector<std::vector<Dtype> > detect_face(cv::Mat& image) {
		//The first 4 elements in all_boxes is the bounding box, and the 5th is probability
		//Attent the depth has changed...
//        input_channels_ = 2;
		cv::Mat img(image.rows, image.cols, image.depth());
		image.copyTo(img);
		cv::transpose(img, img);
		std::vector<std::vector<Dtype> > all_boxes;
//        double t = (double)cvGetTickCount();
		all_boxes = propose_bboxes(img, 0.709, 0.7);
////        t = (double)cvGetTickCount() - t;
////        printf( "run time = %gms\n", t/(cvGetTickFrequency()*1000));
//////        display_faces(img, all_boxes, "pnet", false);
////        t = (double)cvGetTickCount();
//		all_boxes = refine_bboxes(img, all_boxes, 0.6);
//////        t = (double)cvGetTickCount() - t;
//////        printf( "run time = %gms\n", t/(cvGetTickFrequency()*1000));
////////        display_faces(img, all_boxes, "rnet", false);
//////        t = (double)cvGetTickCount();
////		all_boxes = output_bboxes(img, all_boxes, 0.6);
//////        t = (double)cvGetTickCount() - t;
//////        printf( "run time = %gms\n", t/(cvGetTickFrequency()*1000));
////////        display_faces(img, all_boxes, "onet", true);
//////////	  	// return alignment_faces(image, all_boxes);
		return all_boxes;
	}
private:
	//Make a vector of Blob for input an image
	std::vector<caffe::Blob<Dtype>*> prepare_data(cv::Mat& img, std::vector<Bbox> bboxes, int height, int width, int channels, caffe::Blob<Dtype> &input_blob) {
		std::vector<cv::Mat> img_vec;
		for(size_t iter=0;iter<bboxes.size();iter++){
			cv::Mat img_n(height, width, img.depth());
			cv::Rect facebbx(bboxes[iter].y1, bboxes[iter].x1, bboxes[iter].y2 - bboxes[iter].y1+1, bboxes[iter].x2 - bboxes[iter].x1+1);
			cv::Mat tmp_img = img(facebbx);
			tmp_img.convertTo(tmp_img, img.depth());
			cv::resize(tmp_img, img_n, img_n.size(), 0, 0, cv::INTER_AREA);
            img_vec.push_back(img_n);
		}
		int num_patches = bboxes.size();
		input_blob.Reshape(num_patches, channels, height, width);
		transformer_->Transform(img_vec, &input_blob);
		std::vector<caffe::Blob<Dtype>*> input_data;
		input_data.push_back(&input_blob);
		return input_data;
	}
	void modify_network_input(boost::shared_ptr<caffe::Net<Dtype> > &neuralnetwork, int num_patches, int channels, int height, int width) {
		int layer_id = 0;
		int top_id = neuralnetwork->top_ids(layer_id)[0];
		std::vector<caffe::Blob<Dtype>*> top_vec = neuralnetwork->top_vecs()[top_id];
		caffe::Blob<Dtype>* blob_ptr = top_vec[0];
		blob_ptr->Reshape(num_patches, channels, height, width);
	}
	//Pass the image through PNet
	std::vector<std::vector<Dtype> > propose_bboxes(cv::Mat& img, const float down_factor, const float threshold) {
		int height = img.rows;
		int width = img.cols;
        float cell_size = 12.0;
		float down_m = cell_size/50.0;
		float min_size = float(std::min(height, width))*down_m;
		std::vector<float> scales;
		int down_times = 0;
		while(min_size>=cell_size) {
			scales.push_back(down_m*std::pow(down_factor, down_times));
			min_size = min_size*down_factor;
			down_times++;
		}

		cv::Mat img_n;
		std::vector<Bbox> box_img;
		std::vector<std::vector<Dtype> > all_bboxes;
		DetectTools dt_tools;
        int channels = input_channels_;
		for(size_t iter = 0;iter<scales.size();iter++) {
			int hs = int(std::ceil(float(height*scales[iter])));
			int ws = int(std::ceil(float(width*scales[iter])));
			std::vector<caffe::Blob<Dtype>*> input_data;
			input_data.clear();
			box_img.clear();
			box_img.push_back(Bbox(0, 0, height-1, width-1));
			caffe::Blob<Dtype> input_blob;
			input_data = prepare_data(img, box_img, hs, ws, channels, input_blob);

			modify_network_input(PNet_, 1, channels, hs, ws);
			std::vector<caffe::Blob<Dtype>*> output_data	 = PNet_->Forward(input_data);
//            std::printf("Input size: %d, %d\n", hs, ws);
			std::vector<std::vector<Dtype> > bboxes_info = blob2vector(output_data, threshold, scales[iter], 1);

			std::vector< std::vector<Dtype> > boxes_valid_ = dt_tools.generateBboxes(bboxes_info, cell_size);
			dt_tools.nonmaximumSuppression(boxes_valid_, 0.5, dt_tools.UNION);
			all_bboxes.insert(all_bboxes.end(), boxes_valid_.begin(), boxes_valid_.end());
		}
		dt_tools.nonmaximumSuppression(all_bboxes, 0.7, dt_tools.UNION);
		std::vector<std::vector<Dtype> > selected_bboxes = dt_tools.caliberateBboxes(all_bboxes);
	  	dt_tools.turn2rect(selected_bboxes);

		return selected_bboxes;
	}
	std::vector<std::vector<Dtype> > refine_bboxes(cv::Mat&img, std::vector<std::vector<Dtype> > &all_bboxes, float threshold) {
        std::vector<std::vector<Dtype> > selected_bboxes;
        if(all_bboxes.size() == 0) return selected_bboxes;
		//First of all, we will prepare the input data for RNet
	  	DetectTools dt_tools;
	  	std::vector<std::vector<Dtype> > img_patches = dt_tools.bboxes2patches(all_bboxes, img.rows, img.cols);
		std::vector<Bbox> box_img;
		int num_boxes = img_patches.size();
		int hs = 24;
		int ws = 24;
        int channels = input_channels_;
		for(size_t iter = 0;iter<num_boxes;iter++) {
			box_img.push_back(Bbox(img_patches[iter][4], img_patches[iter][5], img_patches[iter][6], img_patches[iter][7]));
		}

		//Transform the Mat file into Blob
		caffe::Blob<Dtype> input_blob;
		std::vector<caffe::Blob<Dtype>*> input_data = prepare_data(img, box_img, hs, ws, channels, input_blob);
		modify_network_input(RNet_, num_boxes, channels, hs, ws);
		std::vector<caffe::Blob<Dtype>*> output_data = RNet_->Forward(input_data);
		std::vector<std::vector<Dtype> > bboxes_info = blob2vector(output_data, 0.0, 1.0, 2);
		std::vector<std::vector<Dtype> > selected_bboxes_mv;
		for(size_t iter = 0;iter<num_boxes;iter++) {
			if(bboxes_info[iter][0]>threshold) {
				std::vector<Dtype> boxandmv;
				boxandmv.push_back(all_bboxes[iter][0]);
				boxandmv.push_back(all_bboxes[iter][1]);
				boxandmv.push_back(all_bboxes[iter][2]);
				boxandmv.push_back(all_bboxes[iter][3]);
				boxandmv.push_back(all_bboxes[iter][4]);
				boxandmv.push_back(all_bboxes[iter][5]);

				boxandmv.push_back(bboxes_info[iter][1]);
				boxandmv.push_back(bboxes_info[iter][2]);
				boxandmv.push_back(bboxes_info[iter][3]);
				boxandmv.push_back(bboxes_info[iter][4]);
				selected_bboxes_mv.push_back(boxandmv);
			}
		}
		dt_tools.nonmaximumSuppression(selected_bboxes_mv, 0.7, dt_tools.UNION);
		selected_bboxes = dt_tools.caliberateBboxes(selected_bboxes_mv);
	  	dt_tools.turn2rect(selected_bboxes);

		return selected_bboxes;
	}

	std::vector<std::vector<Dtype> > output_bboxes(cv::Mat&img, std::vector<std::vector<Dtype> > &all_bboxes, float threshold) {
        std::vector<std::vector<Dtype> > selected_bboxes;
        if(all_bboxes.size() == 0) return selected_bboxes;
	  	DetectTools dt_tools;
	  	std::vector<std::vector<Dtype> > img_patches = dt_tools.bboxes2patches(all_bboxes, img.rows, img.cols);
		std::vector<Bbox> box_img;
		int num_boxes = img_patches.size();
		int hs = 48;
		int ws = 48;
		// num_boxes = 2;
		for(size_t iter = 0;iter<num_boxes;iter++) {
			box_img.push_back(Bbox(img_patches[iter][4], img_patches[iter][5], img_patches[iter][6], img_patches[iter][7]));
		}

		//Transform the Mat file into Blob
		caffe::Blob<Dtype> input_blob;
        int channels = input_channels_;
		std::vector<caffe::Blob<Dtype>*> input_data = prepare_data(img, box_img, hs, ws, channels, input_blob);
		modify_network_input(ONet_, num_boxes, channels, hs, ws);
		std::vector<caffe::Blob<Dtype>*> output_data = ONet_->Forward(input_data);
		std::vector<std::vector<Dtype> > bboxes_info = blob2vector(output_data, 0.0, 1.0, 2);
		std::vector<std::vector<Dtype> > selected_bboxes_mv;
//        std::printf("The number of onet input is %d\n", num_boxes);
		for(size_t iter = 0;iter<num_boxes;iter++) {
			if(bboxes_info[iter][0]>threshold) {
				all_bboxes[iter][4] = bboxes_info[iter][0];
				std::vector<Dtype> boxandmv = all_bboxes[iter];
				boxandmv.push_back(bboxes_info[iter][1]);//6
				boxandmv.push_back(bboxes_info[iter][2]);
				boxandmv.push_back(bboxes_info[iter][3]);
				boxandmv.push_back(bboxes_info[iter][4]);//9
				for(int pt_id = 0;pt_id<10;pt_id++) {
					boxandmv.push_back(bboxes_info[iter][pt_id+8]);
				}
				selected_bboxes_mv.push_back(boxandmv);
			}
		}
		dt_tools.nonmaximumSuppression(selected_bboxes_mv, 0.7, dt_tools.MIN);
        selected_bboxes = dt_tools.caliberateBboxes(selected_bboxes_mv);
		return selected_bboxes;
	}

	std::vector<std::vector<Dtype> > tracking_bboxes(cv::Mat&img, std::vector<std::vector<Dtype> > &all_bboxes, float threshold) {
		std::vector<std::vector<Dtype> > selected_bboxes;
		if(all_bboxes.size() == 0) return selected_bboxes;
		DetectTools dt_tools;
		std::vector<std::vector<Dtype> > img_patches = dt_tools.bboxes2patches(all_bboxes, img.rows, img.cols);
		std::vector<Bbox> box_img;
		int num_boxes = img_patches.size();
		int hs = 32;
		int ws = 32;
		// num_boxes = 2;
		for(size_t iter = 0;iter<num_boxes;iter++) {
			box_img.push_back(Bbox(img_patches[iter][4], img_patches[iter][5], img_patches[iter][6], img_patches[iter][7]));
		}

		//Transform the Mat file into Blob
		caffe::Blob<Dtype> input_blob;
		int channels = input_channels_;
		std::vector<caffe::Blob<Dtype>*> input_data = prepare_data(img, box_img, hs, ws, channels, input_blob);
		modify_network_input(TNet_, num_boxes, channels, hs, ws);
		std::vector<caffe::Blob<Dtype>*> output_data = TNet_->Forward(input_data);
		std::vector<std::vector<Dtype> > bboxes_info = blob2vector(output_data, 0.0, 1.0, 4);
		std::vector<std::vector<Dtype> > selected_bboxes_mv;
		all_bboxes[0][4] = bboxes_info[0][0];
		std::vector<Dtype> boxandmv = all_bboxes[0];
        boxandmv.push_back(bboxes_info[0][1]);//6
		boxandmv.push_back(bboxes_info[0][2]);
		boxandmv.push_back(bboxes_info[0][3]);
		boxandmv.push_back(bboxes_info[0][4]);//9
		selected_bboxes_mv.push_back(boxandmv);
		selected_bboxes = dt_tools.caliberateBboxes(selected_bboxes_mv);

		return selected_bboxes;
	}

	std::vector<std::vector<Dtype> > blob2vector(std::vector<caffe::Blob<Dtype>*>& output_data, float threshold, float scale, int whichnet) {
		caffe::Blob<Dtype>* output_blob_ptr_bboxregression = output_data[0];
		caffe::Blob<Dtype>* output_blob_ptr_probability;
		caffe::Blob<Dtype>* output_blob_ptr_landmarks;
		if(whichnet == 3) output_blob_ptr_probability = output_data[2];
		else output_blob_ptr_probability = output_data[1];
		if(whichnet == 3) output_blob_ptr_landmarks = output_data[1];
		int num_patches_out = output_blob_ptr_probability->num();
	  	int num_channels_out = output_blob_ptr_probability->channels();
	  	int height_out = output_blob_ptr_probability->height();
		int width_out = output_blob_ptr_probability->width();
//        std::printf("p: %d, c:%d, h:%d, w:%d\n", num_patches_out, num_channels_out, height_out, width_out);
		int num_patches_out_r = output_blob_ptr_bboxregression->num();
	  	int num_channels_out_r = output_blob_ptr_bboxregression->channels();
	  	int height_out_r = output_blob_ptr_bboxregression->height();
		int width_out_r = output_blob_ptr_bboxregression->width();
//        std::printf("p: %d, c:%d, h:%d, w:%d\n", num_patches_out_r, num_channels_out_r, height_out_r, width_out_r);
		const Dtype *ptr_prob = output_blob_ptr_probability->cpu_data();
	  	const Dtype *ptr_bxrg = output_blob_ptr_bboxregression->cpu_data();
		const Dtype *ptr_ldmk;
	  	if(whichnet == 3) ptr_ldmk = output_blob_ptr_landmarks->cpu_data();
	  	int num_patches_out_l = 0;
	  	int num_channels_out_l = 0;
	  	int height_out_l = 0;
	  	int width_out_l = 0;
	  	if(whichnet ==3) {
	  		num_patches_out_l = output_blob_ptr_landmarks->num();
	  		num_channels_out_l = output_blob_ptr_landmarks->channels();
	  		height_out_l = output_blob_ptr_landmarks->height();
	  		width_out_l = output_blob_ptr_landmarks->width();
	  	}
	  	std::vector<std::vector<Dtype> > bboxes_info;
		int valid = 0;
	  	for(int ind_n = 0;ind_n<num_patches_out;ind_n++) {
	  		for(int ind_c = 1;ind_c<num_channels_out;ind_c++) {
		  		for(int ind_h = 0;ind_h<height_out;ind_h++) {
		  			for(int ind_w = 0;ind_w<width_out;ind_w++) {
		  				Dtype point_v;
			  			point_v = *(ptr_prob+ ( (ind_n*num_channels_out + ind_c)*height_out +ind_h)*width_out + ind_w);
//                        if(whichnet == 4) {
//                            float tmp = *(ptr_prob+ ( (ind_n*num_channels_out + 0)*height_out +ind_h)*width_out + ind_w);
//                            std::printf("%f VS %f\n", point_v, tmp);
//                        }
						if(point_v>=threshold) {
	  						valid++;
		  					std::vector<Dtype> bbox_information;
		  					bbox_information.push_back(point_v);//0
		  					point_v = *(ptr_bxrg+ ( (ind_n*num_channels_out_r + 0)*height_out_r +ind_h)*width_out_r + ind_w );
//							if(whichnet == 4) std::printf("%f ", point_v);
		 					bbox_information.push_back(point_v);//1
		 					point_v = *(ptr_bxrg+ ( (ind_n*num_channels_out_r + 1)*height_out_r +ind_h)*width_out_r + ind_w );
//							if(whichnet == 4) std::printf("%f ", point_v);
	  						bbox_information.push_back(point_v);//2
	 						point_v = *(ptr_bxrg+ ( (ind_n*num_channels_out_r + 2)*height_out_r +ind_h)*width_out_r + ind_w );
//							if(whichnet == 4) std::printf("%f ", point_v);
	  						bbox_information.push_back(point_v);//3
		  					point_v = *(ptr_bxrg+ ( (ind_n*num_channels_out_r + 3)*height_out_r +ind_h)*width_out_r + ind_w );
//							if(whichnet == 4) std::printf("%f\n", point_v);
		  					bbox_information.push_back(point_v);//4
		  					bbox_information.push_back(float(ind_h+1));//5
		  					bbox_information.push_back(float(ind_w+1));//6
		  					bbox_information.push_back(scale);//7
		  					if(whichnet == 3) {
		  						for(int pt_id=0;pt_id<10;pt_id++) {
		  							point_v = *(ptr_ldmk+ ( (ind_n*num_channels_out_l + pt_id)*height_out_l +ind_h)*width_out_l + ind_w );
		  							bbox_information.push_back(point_v);
		  							// printf("%f ", point_v);
		  						}
		  						// printf("\n");
		  					}
		  					bboxes_info.push_back(bbox_information);
		  				}
		  			}
		  		}
			}
	  	}
	  	return bboxes_info;
	}

	void print_blob(std::vector<caffe::Blob<Dtype>*> &input_data) {
		caffe::Blob<Dtype>* input_blob_in = input_data[0];
		int num_patches_in = input_blob_in->num();
	  	int num_channels_in = input_blob_in->channels();
	  	int height_in = input_blob_in->height();
		int width_in = input_blob_in->width();
		const Dtype *ptr_prob = input_blob_in->cpu_data();
	  	for(int ind_n = 0;ind_n<num_patches_in;ind_n++) {
	  		for(int ind_c = 1;ind_c<num_channels_in;ind_c++) {
		  		for(int ind_h = 0;ind_h<height_in;ind_h++) {
		  			for(int ind_w = 0;ind_w<width_in;ind_w++) {
		  				Dtype point_v;
			  			point_v = *(ptr_prob+ ( (ind_n*num_channels_in + ind_c)*height_in +ind_h)*width_in + ind_w);
		  				printf("%f ", point_v);
		  			}
		  			printf("\n");
		  		}
		  	}
		}
	}
	void display_faces(cv::Mat&img, std::vector<std::vector<Dtype> > bboxes, std::string window_name, bool landmarks) {
		cv::Mat image(img.size(), CV_8UC1);
		img.copyTo(image);
		for(size_t iter = 0;iter<bboxes.size();iter++) {
			cv::rectangle(image, cv::Point(bboxes[iter][1], bboxes[iter][0]), cv::Point(bboxes[iter][3], bboxes[iter][2]), cv::Scalar(255, 255, 255));
			if(landmarks && bboxes[iter].size()==16){
				for(int jter = 0;jter<5;jter++) {
					cv::drawMarker(image, cv::Point(bboxes[iter][jter*2 + 7], bboxes[iter][jter*2 + 6]), cv::Scalar(0, 255, 0), cv::MARKER_STAR, 4, 2);
				}
			}
		}
//		cv::cvtColor(image, image, CV_BGR2RGB);
		cv::transpose(image, image);
		cv::imshow(window_name, image);
		cv::waitKey(0);
	}
private:
	boost::shared_ptr<caffe::Net<Dtype> > PNet_;
	boost::shared_ptr<caffe::Net<Dtype> > RNet_;
	boost::shared_ptr<caffe::Net<Dtype> > ONet_;
	boost::shared_ptr<caffe::Net<Dtype> > TNet_;
	boost::shared_ptr<caffe::DataTransformer<Dtype> > transformer_;
    int input_channels_;
	bool has_tnet_;
};
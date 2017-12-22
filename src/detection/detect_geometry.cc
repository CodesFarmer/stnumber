#include "detect_geometry.h"

std::vector< std::vector<Ntype> > DetectTools::generateBboxes(std::vector<std::vector<Ntype> > boxes_probability, float cellsize) {
//	float cellsize = 12;
	float stride = 2;

	std::vector< std::vector<Ntype> > bboxes;
	BBXINFO boxandinfo;
	boxandinfo.resize(10);

	for(size_t iter=0;iter<boxes_probability.size();iter++) {
		//The probability belong to a face is beyond the threshold
		float scale = boxes_probability[iter][7];
		boxandinfo[0] = std::floor( ( (boxes_probability[iter][5]-1) *stride+1)/scale);
		boxandinfo[1] = std::floor( ( (boxes_probability[iter][6]-1) *stride+1)/scale);
		boxandinfo[2] = std::floor( ( (boxes_probability[iter][5]-1) *stride+cellsize)/scale);
		boxandinfo[3] = std::floor( ( (boxes_probability[iter][6]-1) *stride+cellsize)/scale);
		boxandinfo[4] = boxes_probability[iter][0];
		boxandinfo[5] = (boxandinfo[2]-boxandinfo[0]+1)*(boxandinfo[3]-boxandinfo[1]+1);
		boxandinfo[6] = boxes_probability[iter][1];
		boxandinfo[7] = boxes_probability[iter][2];
		boxandinfo[8] = boxes_probability[iter][3];
		boxandinfo[9] = boxes_probability[iter][4];
		bboxes.push_back(boxandinfo);
	}

	return bboxes;
}

void DetectTools::nonmaximumSuppression(std::vector<std::vector<Ntype> >& bboxes, float threshold, NMSType nmstype_) {
	//sort the bounding box by probability from large to low
	std::sort(bboxes.begin(), bboxes.end(), isLarger());
	std::vector<std::vector<Ntype> >::iterator iter;
	std::vector<std::vector<Ntype> >::iterator jter;
	for(iter=bboxes.begin();iter<bboxes.end();iter++) {
		for(jter=iter+1;jter<bboxes.end();jter++){
			std::vector<Ntype> elems_i = *(iter);
			std::vector<Ntype> elems_j = *(jter);
			float x1_inter = std::max(elems_i[0], elems_j[0]);
			float y1_inter = std::max(elems_i[1], elems_j[1]);
			float x2_inter = std::min(elems_i[2], elems_j[2]);
			float y2_inter = std::min(elems_i[3], elems_j[3]);
			float w_inter = std::max(0.0, double(x2_inter - x1_inter+1) );
			float h_inter = std::max(0.0, double(y2_inter - y1_inter+1) );
			float area_inter = 1.0;
			if(nmstype_ == UNION) {
				area_inter = (w_inter*h_inter)/(elems_i[5] + elems_j[5] - w_inter*h_inter);
			}
			else{
				area_inter = (w_inter*h_inter)/std::min(elems_i[5], elems_j[5]);
			}
			//delete the bounding boxes whose overlop between iter is larger than threshold
			if(area_inter > threshold) {
				jter = bboxes.erase(jter);
				jter = jter - 1;
			}
		}
	}
}

void DetectTools::turn2rect(std::vector<std::vector<Ntype> >& bboxes) {
	Ntype x,y,ex,ey;
	for(size_t iter = 0;iter<bboxes.size();iter++) {
		Ntype h_b = bboxes[iter][2] - bboxes[iter][0] + 1;
		Ntype w_b = bboxes[iter][3] - bboxes[iter][1] + 1;
		Ntype maxLen = std::max(w_b, h_b);
		x = bboxes[iter][0] + h_b*0.5 - maxLen*0.5;
		y = bboxes[iter][1] + w_b*0.5 - maxLen*0.5;
		ex = x + maxLen;
		ey = y + maxLen;
		bboxes[iter][0] = x;
		bboxes[iter][1] = y;
		bboxes[iter][2] = ex;
		bboxes[iter][3] = ey;
		bboxes[iter][5] = (ex-x+1)*(ey-y+1);
		// printf("%f %f %f %f %f\n", x, y, ex, ey, bboxes[iter][4]);
	}
}

std::vector<std::vector<Ntype> > DetectTools::bboxes2patches(std::vector<std::vector<Ntype> >& bboxes, Ntype height, Ntype width) {
	height = height - 1;
	width = width - 1;
	std::vector<std::vector<Ntype> > image2patch;
	std::vector<Ntype> patch_vec(10);
	Ntype x,y,ex,ey,dx,dy,edx,edy;
	for(size_t iter = 0;iter<bboxes.size();iter++) {
		Ntype w_b = bboxes[iter][2] - bboxes[iter][0] + 1;
		Ntype h_b = bboxes[iter][3] - bboxes[iter][1] + 1;
		x = bboxes[iter][0];
		y = bboxes[iter][1];
		ex = bboxes[iter][2];
		ey = bboxes[iter][3];
		dx = 0;
		dy = 0;
		edx = w_b-1;
		edy = h_b-1;
		if(ex>height){
			edx = -ex + height + h_b;
			ex = height;
		}
		if(ey>width){
			edy = -ey + width + w_b;
			ey = width;
		}
		if(x<0){
			dx = 1-x;
			x = 0;
		}
		if(y<0){
			dy = 1-y;
			y = 0;
		}
		patch_vec[0] = dx;
		patch_vec[1] = edx;
		patch_vec[2] = dy;
		patch_vec[3] = edy;
		patch_vec[4] = x;
		patch_vec[5] = y;
		patch_vec[6] = ex-1;
		patch_vec[7] = ey-1;
		patch_vec[8] = w_b;
		patch_vec[9] = h_b;

		image2patch.push_back(patch_vec);
	}
	return image2patch;
}

std::vector<std::vector<Ntype> > DetectTools::caliberateBboxes(std::vector<std::vector<Ntype> >& bboxes_mv) {
	//The function will adjust the boundary of bounding boxes
	//The regression information are stored at bboxes_mv[6-9]
	Ntype h_b;
	Ntype w_b;
	std::vector<std::vector<Ntype> > adjusted_bboxes;
	for(size_t iter=0;iter < bboxes_mv.size();iter++) {
		h_b = bboxes_mv[iter][2] - bboxes_mv[iter][0];
		w_b = bboxes_mv[iter][3] - bboxes_mv[iter][1];
		std::vector<Ntype> bbox;
		bbox.push_back(bboxes_mv[iter][0] + bboxes_mv[iter][6]*h_b);
		bbox.push_back(bboxes_mv[iter][1] + bboxes_mv[iter][7]*w_b);
		bbox.push_back(bboxes_mv[iter][2] + bboxes_mv[iter][8]*h_b);
		bbox.push_back(bboxes_mv[iter][3] + bboxes_mv[iter][9]*w_b);
		bbox.push_back(bboxes_mv[iter][4]);
		bbox.push_back( (bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1) );
		if(bbox[0] < bbox[2] && bbox[1] < bbox[3]) {
			adjusted_bboxes.push_back(bbox);
		}
	}
	return adjusted_bboxes;
}
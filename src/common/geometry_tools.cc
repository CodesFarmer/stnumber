#include "geometry_tools.h"

namespace GEOMETRYTOOLS{
    float regionsIOU(std::vector<cv::Rect>& rects, cv::Rect the_rect) {
        float max_iou = 0.0f;
        for(size_t iter = 0; iter < rects.size(); iter++) {
            float iou = regionIOU(rects[iter], the_rect);
            if(iou > max_iou) max_iou = iou;
        }
        return max_iou;
    }
    float regionIOU(cv::Rect& rect1, cv::Rect& rect2) {
        float x1 = std::max(rect1.x, rect2.x);
        float y1 = std::max(rect1.y, rect2.y);
        float x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
        float y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
        float width = std::max(0.0f, x2 - x1);
        float height = std::max(0.0f, y2 - y1);
        float area_inter = width*height;
        float area1 = rect1.width*rect1.height;
        float area2 = rect2.width*rect2.height;
        return area_inter/(area1 + area2 - area_inter);
    }
}
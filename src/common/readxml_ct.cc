#include "readxml_ct.h"


std::vector<cv::Rect> get_bounding_boxes(std::string file_xml) {
    std::vector<cv::Rect> img_rect;
    if(access(file_xml.c_str(), F_OK) == -1){
        return img_rect;
    }
    tinyxml2::XMLDocument doc;
    if(doc.LoadFile(file_xml.c_str()) != 0) {
        return img_rect;
    }
    tinyxml2::XMLElement *object = doc.FirstChildElement("annotation")->FirstChildElement("object");
    //->FirstChildElement("polygon")->FirstChildElement("pt");

    while(object != NULL) {
        tinyxml2::XMLElement *point = object->FirstChildElement("polygon")->FirstChildElement("pt");
        cv::Rect rect;
        bool valid = true;
        if(point != NULL) {
            const char *pt = point->FirstChildElement("x")->GetText();
            rect.x = std::atoi(pt);
            pt = point->FirstChildElement("y")->GetText();
            rect.y = std::atoi(pt);
            point = point->NextSiblingElement("pt");
            if(point != NULL) point = point->NextSiblingElement("pt");
            else valid = false;
            if(point != NULL) {
                pt = point->FirstChildElement("x")->GetText();
                int x2 = std::atoi(pt);
                pt = point->FirstChildElement("y")->GetText();
                int y2 = std::atoi(pt);
                int x1 = std::min(rect.x, x2);
                int y1 = std::min(rect.y, y2);
                rect.width = std::abs(x2 - rect.x);
                rect.height = std::abs(y2 - rect.y);
                rect.x = x1;
                rect.y = y1;
            }
            else valid = false;
        }
        else valid = false;
        if(valid) img_rect.push_back(rect);
        object = object->NextSiblingElement("object");
    }
    return img_rect;
}
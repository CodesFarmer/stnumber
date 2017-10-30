#include "readxml_ct.h"


std::vector<cv::Rect> get_bounding_boxes(std::string file_xml) {
    tinyxml2::XMLDocument doc;
    doc.LoadFile(file_xml.c_str());
    tinyxml2::XMLElement *object = doc.FirstChildElement("annotation")->FirstChildElement("object");
    //->FirstChildElement("polygon")->FirstChildElement("pt");

    std::vector<cv::Rect> img_rect;
    while(object != NULL) {
        tinyxml2::XMLElement *point = object->FirstChildElement("polygon")->FirstChildElement("pt");
        cv::Rect rect;
        const char *pt = point->FirstChildElement("x")->GetText();
        rect.x = std::atoi(pt);
        pt = point->FirstChildElement("y")->GetText();
        rect.y = std::atoi(pt);
        point = point->NextSiblingElement("pt");
        point = point->NextSiblingElement("pt");
        pt = point->FirstChildElement("x")->GetText();
        rect.width = std::atoi(pt) - rect.x;
        pt = point->FirstChildElement("y")->GetText();
        rect.height = std::atoi(pt) - rect.y;
        img_rect.push_back(rect);
        object = object->NextSiblingElement("object");
    }
    return img_rect;
}
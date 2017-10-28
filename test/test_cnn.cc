#include <opencv2/opencv.hpp>
#include <iostream>

#include "retrieve_output.h"
#include "filepath.h"
#include "readxml_ct.h"

int main(int argv, char * argc[]) {
    tinyxml2::XMLDocument doc;
    doc.LoadFile("/home/slam/datasets/handdetect_sample/annotations/1509071298190268772.xml");
    tinyxml2::XMLElement *object = doc.FirstChildElement("annotation")->FirstChildElement("object")->FirstChildElement("polygon");
    const char *title = object->FirstChildElement("username")->GetText();
    std::cout<<title<<std::endl;
    return 0;
}
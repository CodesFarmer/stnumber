#include <iostream>
#include <string>

#include "filepath.h"

int main() {
    std::string fullpath = "name.test";
    std::string file_path;
    std::string file_name;
    std::string file_ext;
    FILEPARTS::fileparts(fullpath, file_path, file_name, file_ext);
    std::cout<<"File Path: "<<file_path<<std::endl;
    std::cout<<"File Name: "<<file_name<<std::endl;
    std::cout<<"File Ext: "<<file_ext<<std::endl;
    return 0;
}
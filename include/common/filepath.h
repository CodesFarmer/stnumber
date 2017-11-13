//
// Created by slam on 17-10-26.
//
/*
 * This file will split the full file path into three parts
 * 1. The route to the file
 * 2. The name of the file
 * 3. The format of the file
 * =======================================================
 * */

#ifndef PROJECT_FILEPATH_H
#define PROJECT_FILEPATH_H

#include <string>
#include <iostream>
#include <fstream>

namespace FILEPARTS{
    bool fileparts(std::string, std::string &, std::string &, std::string&);
    bool replace_string(std::string &, const std::string &, const std::string &);
    int counting_lines(std::string);
}

#endif //PROJECT_FILEPATH_H

#include "filepath.h"

namespace FILEPARTS {
    bool fileparts(std::string fullpath, std::string & route_path, std::string &file_name, std::string &file_ext) {
        int str_len = int(fullpath.size());
        int pos1;
        if((pos1 = fullpath.rfind('.')) == std::string::npos) {
            std::cerr<<"The file does not have an extension..."<<std::endl;
            return false;
        }
        file_ext = fullpath.substr(pos1 + 1, str_len - pos1);
        int pos2;
        if((pos2 = fullpath.rfind('/')) == std::string::npos) {
            pos2 = -1;
        }
        file_name = fullpath.substr(pos2 + 1, pos1 - 1 - pos2);
        if(pos2 > -1) {
            route_path = fullpath.substr(0, pos2);
        }
        else {
            route_path = "";
        }
        return true;
    }
}
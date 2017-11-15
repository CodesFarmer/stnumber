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
    int counting_lines(std::string filename) {
        std::ifstream input_fid;
        input_fid.open(filename.c_str(), std::ios::in|std::ios::binary);
        int num_lines = 0;
        std::string lines;
        while(std::getline(input_fid, lines))
            num_lines++;
        return num_lines;
    }

    bool replace_string(std::string &str, const std::string &substr_1, const std::string &substr_2) {
        size_t start_pos = str.find(substr_1);
        if(start_pos == std::string::npos) return false;
        str.replace(start_pos, substr_1.length(), substr_2);
        return true;
    }

    bool fullfile(std::string &file_path, int nargs, ...) {
        va_list ap;
        va_start(ap, nargs);
        file_path = va_arg(ap, std::string);
        for(int i = 1; i<nargs; i++) {
            file_path = file_path + "/" + va_arg(ap, std::string);
        }
        va_end(ap);
        replace_string(file_path, "//", "/");
    }
}
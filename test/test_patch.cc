#include "generate_patch.h"
#include <boost/shared_ptr.hpp>

int main(int argc, char *argv[]) {
    assert(argc == 3);
    std::string file_lists(argv[1]);
    std::string dest_path(argv[2]);
    boost::shared_ptr<GeneratePatch> generator(new GeneratePatch(50, 50));
    std::cout<<"Test 1... "<<std::endl;
    generator->generate_patches(file_lists, 12, dest_path);

    return 0;
}
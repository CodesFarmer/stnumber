#include "generate_patch.h"
#include <boost/shared_ptr.hpp>

int main(int argc, char *argv[]) {
//    assert(argc == 3);
//    std::string file_lists(argv[1]);
//    std::string dest_path(argv[2]);
//    boost::shared_ptr<GeneratePatch> generator(new GeneratePatch());
//    std::cout<<"Test 1... "<<std::endl;
//    generator->generate_patches(file_lists, 12, dest_path);

    assert(argc == 3);
    std::string file_lists(argv[1]);
    std::string dest_path(argv[2]);
    boost::shared_ptr<GeneratePatch> generator(new GeneratePatch(25, 25));
    std::map<std::string, std::pair<std::string, std::string> > modelpath;
    std::string models_dir = "../data/model/";
    modelpath["pnet"] = std::make_pair(models_dir + std::string("pnet.caffemodel"), models_dir+std::string("pnet_deploy.prototxt"));
    modelpath["rnet"] = std::make_pair(models_dir + std::string("rnet.caffemodel"), models_dir+std::string("rnet_deploy.prototxt"));
    std::vector<float> mean_value(1, 17.2196);
    float img2net_scale = 0.0125;
    generator->initialize_detector(modelpath, img2net_scale, mean_value);
    std::printf("TSTESTSTSDSTSTS\n");
    generator->generate_patches_cnn(file_lists, 48, dest_path);

    return 0;
}
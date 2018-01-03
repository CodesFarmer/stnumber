#include "generate_patch.h"
#include <boost/shared_ptr.hpp>

int main(int argc, char *argv[]) {
//    assert(argc == 3);
//    std::string file_lists(argv[1]);
//    std::string dest_path(argv[2]);
//    boost::shared_ptr<GeneratePatch> generator(new GeneratePatch(30, 20));
//    std::cout<<"Processing... "<<std::endl;
//    generator->generate_patches_crop(file_lists, 12, dest_path, GeneratePatch::HDF5);

    assert(argc == 3);
    std::string file_lists(argv[1]);
    std::string dest_path(argv[2]);
    boost::shared_ptr<GeneratePatch> generator(new GeneratePatch(2, 2));
    std::map<std::string, std::pair<std::string, std::string> > modelpath;
    std::string models_dir = "data/model/";
    modelpath["pnet"] = std::make_pair(models_dir + std::string("pnet.caffemodel"), models_dir+std::string("pnet_deploy.prototxt"));
    modelpath["rnet"] = std::make_pair(models_dir + std::string("rnet.caffemodel"), models_dir+std::string("rnet_deploy.prototxt"));
    modelpath["onet"] = std::make_pair(models_dir + std::string("onet.caffemodel"), models_dir+std::string("onet_deploy.prototxt"));
    int channels = 1;
#ifdef USE_DEPTH
    channels = 2;
#endif
    std::vector<float> mean_value(channels, 0.0);
    float img2net_scale = 1.0f;
    generator->initialize_detector(modelpath, img2net_scale, mean_value);
    std::cout<<"Processing... "<<std::endl;
    generator->generate_patches_cnn(file_lists, 48, dest_path, GeneratePatch::HDF5);


//    assert(argc == 3);
//    std::string file_lists(argv[1]);
//    std::string dest_path(argv[2]);
//    boost::shared_ptr<GeneratePatch> generator(new GeneratePatch(2, 2));
//    std::cout<<"Processing... "<<std::endl;
//    generator->generate_patches_text(file_lists, 48, dest_path, GeneratePatch::HDF5);

    return 0;
}
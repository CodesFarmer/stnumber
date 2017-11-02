CaffePath="/home/public/nfs70/rnn_sh/caffe-master/build/tools"
${CaffePath}/caffe train -solver prototxt/rnet/rnet_cla_solver.prototxt -gpu 1
#${CaffePath}/caffe train -solver prototxt/rnet/rnet_solver.prototxt -gpu 1 -weights model/rnet/rnet_cla_iter_500000.caffemodel

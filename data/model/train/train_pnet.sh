CaffePath="/home/public/nfs70/rnn_sh/caffe-master/build/tools"
#${CaffePath}/caffe train -solver prototxt/pnet_cla_solver.prototxt -gpu 0
#${CaffePath}/caffe train -solver prototxt/pnet_solver.prototxt -gpu 0 -weights model/pnet/pnet_iter_10000.caffemodel
${CaffePath}/caffe train -solver prototxt/pnet_solver.prototxt -gpu 0

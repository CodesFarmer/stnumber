clear;clc;
addpath('/home/slam/DepthGesture/caffe-master/matlab');
caffe.reset_all();

% load face model and creat network
caffe.set_device(0);
caffe.set_mode_gpu();
model = '../data/model/pnet_deploy.prototxt';
weights = '../data/model/pnet_finetune_iter_3500.caffemodel';
net = caffe.Net(model, weights, 'test');

img=imread('000.png');
img = (img - 17.26)/80;
net.blobs('data').reshape([12 12 1 1]);
out=net.forward({img});
caffe.reset_all()
clc
out{2,1}()
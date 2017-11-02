clear;clc;
addpath('/home/slam/DepthGesture/caffe-master/matlab');
caffe.reset_all();

% load face model and creat network
caffe.set_device(0);
caffe.set_mode_gpu();
% model = '../data/model/pnet_deploy.prototxt';
% weights = '../data/model/pnet_finetune_iter_3500.caffemodel';
% model = '../data/model/rnet_deploy.prototxt';
% weights = '../data/model/rnet_cla_iter_19700.caffemodel';
model = '../data/model/onet_deploy.prototxt';
weights = '../data/model/onet.caffemodel';
net = caffe.Net(model, weights, 'test');

img=imread('301.png');
img = (img - 17.26)/80;
net.blobs('data').reshape([48 48 1 1]);
out=net.forward({img});
caffe.reset_all()
clc
out{2,1}()
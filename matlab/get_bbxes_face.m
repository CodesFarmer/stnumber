clear;clc;
addpath('/home/slam/DepthGesture/caffe-master/matlab');
caffe.reset_all();

% load face model and creat network
caffe.set_device(0);
caffe.set_mode_gpu();
model = 'pnet.prototxt';
weights = 'pnet_iter_100000.caffemodel';
net = caffe.Net(model, weights, 'test');

img=imread('1.jpg');
im_data=(single(img)-127.5)*0.0078125;
% load face image, and align to 112 X 96
net.blobs('data').reshape([12 12 3 1]);
out=net.forward({img});
caffe.reset_all()
clc
out{1,1}()
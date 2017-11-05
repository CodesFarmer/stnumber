img = imread('samples/test_hdf5.png');
scale = 0.0125;
meanval = 17.2196;
% scale = 1;
% meanval = 127.5;

%src_data: w,h,c,n
src_data = zeros(12, 24, 1, 2, 'single');
information = zeros(4, 2, 'single');
information(1,1) = 1;
information(2,1) = 3;
information(3,1) = 2;
information(4,1) = 4;
information(1,2) = 9;
information(2,2) = 8;
information(3,2) = 6;
information(4,2) = 7;
imnorm = (single(img)-meanval)*scale;
src_data(:, :, :, 1) = imnorm;
src_data(:, :, :, 2) = imnorm;

created_flag = false;
filename = 'test.h5';
chunksz = 1;
startloc = struct('dat',[1,1,1,0+1], 'lab', [1,0+1]);
store2hdf5(filename, src_data, information, ~created_flag, startloc, chunksz);
fid = fopen('positive_rand.txt', 'r');

num = 0;
mean_ir = 0;
mean_dp = 0;
var_ir = 0;
var_dp = 0;
while(~feof(fid))
    num = num + 1;
    tline = fgetl(fid);
    fileinfo = strsplit(tline);
    filepath = fileinfo{1,1};
    filename = fileinfo{1,2};
    x = str2double(fileinfo{1,3});
    y = str2double(fileinfo{1,4});
    w = str2double(fileinfo{1,5});
    h = str2double(fileinfo{1,6});
    if(exist(fullfile(filepath, 'cam0', [filename '.png']), 'file') ~= 0)
        im_ir = imread(fullfile(filepath, 'cam0', [filename '.png']));
    elseif(exist(fullfile(filepath, 'cam0', [filename '.jpg']), 'file') ~= 0)
        im_ir = imread(fullfile(filepath, 'cam0', [filename '.jpg']));
    else
        continue;
    end
    if(exist(fullfile(filepath, 'dep0', [filename '.png']), 'file') == 0)
        continue;
    end
    img_ = imread(fullfile(filepath, 'dep0', [filename '.png']));
    im_dp = bitand(img_, 2^13-1);
    
    if(w<0 || h <0)
        continue;
    end
    if(x<0 || y <0)
        continue;
    end
    if(x+w+1>size(im_dp, 2) || y+h+1 > size(im_dp, 1))
        continue;
    end
    
    im_ir_1 = im_ir(y+1:y+h+1, x+1:x+w+1);
    im_dp_1 = im_dp(y+1:y+h+1, x+1:x+w+1);
    mean_ir = mean_ir + mean(im_ir_1(:));
    mean_dp = mean_dp + mean(im_dp_1(:));
    var_ir = var_ir + double(max(im_ir_1(:)));
    var_dp = var_dp + double(max(im_dp_1(:)));
    if(isnan(mean_ir))
        break;
    end
    if(mod(num, 100) == 0)
        fprintf('.');
    end
    if(mod(num, 1000) == 0)
        fprintf('\n');
    end
    if(num > 20000)
        break;
    end
end


fclose(fid);
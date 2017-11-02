function gen_hdf5(filename, filelists1)

fid1 = fopen(fullfile(filelists1, 'positive.txt'), 'r+');
fid2 = fopen(fullfile(filelists1, 'negative.txt'), 'r+');
fid3 = fopen(fullfile(filelists1, 'part.txt'), 'r+');

src_data = zeros(48, 48, 1, 100, 'single');
information = zeros(5, 100, 'single');
num_img = 1;
created_flag = false;
totalct=0;
chunksz=100;
scale = 0.0125;
meanval = 17.2196;

num_all = 0;
while(~feof(fid1) || ~feof(fid2) || ~feof(fid3))
    if(~feof(fid1))
        tline = fgetl(fid1);
        str_parts = strsplit(tline, ' ');
        %read image
        fname = str_parts{1, 1};
        imname = sprintf('%s', fname);
        if(~isempty(strfind(imname, 'cam0')))
            continue;
        end
%         imname = fullfile('/ssd/rnn_sh/detection_tools', imname);
        if(exist(imname))
            img = imread(imname);
            imnorm = (single(img)-meanval)*scale;
            src_data(:, :, :, num_img) = imnorm;
            %read the labels
            label = str2double(str_parts{1, 2});
            if(label == 0)
                roi = [0, 0, 0, 0];
            else
                %read the boundingbox
                roi = str2double(str_parts{1, 3});
                roi = [roi, str2double(str_parts{1, 4})];
                roi = [roi, str2double(str_parts{1, 5})];
                roi = [roi, str2double(str_parts{1, 6})];
            end
	    if(label == -1)
	        label = 2;
	    end
            %write to hdf5
            information(:, num_img) = [label roi]';

            num_img = num_img + 1;

            num_all = num_all + 1;
        end
    end
    if(num_img == chunksz + 1)
      fprintf('.');
      if(mod(num_all, 5000) == 0)
          fprintf('\n');
      end
      startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
      curr_dat_sz = store2hdf5(filename, src_data, information, ~created_flag, startloc, chunksz); 
      created_flag = true;
      totalct = curr_dat_sz(end);
      num_img = 1;
    end
    
    if(~feof(fid2))
        tline = fgetl(fid2);
        str_parts = strsplit(tline, ' ');
        %read image
        fname = str_parts{1, 1};
        imname = sprintf('%s', fname);
        if(~isempty(strfind(imname, 'cam0')))
            continue;
        end
%         imname = fullfile('/ssd/rnn_sh/detection_tools', imname);
        if(exist(imname))
            img = imread(imname);
            imnorm = (single(img)-meanval)*scale;
            src_data(:, :, :, num_img) = imnorm;
            %read the labels
            label = str2double(str_parts{1, 2});
            if(label == 0)
                roi = [0, 0, 0, 0];
            else
                %read the boundingbox
                roi = str2double(str_parts{1, 3});
                roi = [roi, str2double(str_parts{1, 4})];
                roi = [roi, str2double(str_parts{1, 5})];
                roi = [roi, str2double(str_parts{1, 6})];
            end
	    if(label == -1)
	        label = 2;
	    end
            %write to hdf5
            information(:, num_img) = [label roi]';

            num_img = num_img + 1;

            num_all = num_all + 1;
        end
    end
    if(num_img == chunksz + 1)
      fprintf('.');
      if(mod(num_all, 5000) == 0)
          fprintf('\n');
      end
      startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
      curr_dat_sz = store2hdf5(filename, src_data, information, ~created_flag, startloc, chunksz); 
      created_flag = true;
      totalct = curr_dat_sz(end);
      num_img = 1;
    end
    
    if(~feof(fid3))
        tline = fgetl(fid3);
        str_parts = strsplit(tline, ' ');
        %read image
        fname = str_parts{1, 1};
        imname = sprintf('%s', fname);
        if(~isempty(strfind(imname, 'cam0')))
            continue;
        end
%         imname = fullfile('/ssd/rnn_sh/detection_tools', imname);
        
        if(exist(imname))
            img = imread(imname);
            imnorm = (single(img)-meanval)*scale;
            src_data(:, :, :, num_img) = imnorm;
            %read the labels
            label = str2double(str_parts{1, 2});
            if(label == 0)
                roi = [0, 0, 0, 0];
            else
                %read the boundingbox
                roi = str2double(str_parts{1, 3});
                roi = [roi, str2double(str_parts{1, 4})];
                roi = [roi, str2double(str_parts{1, 5})];
                roi = [roi, str2double(str_parts{1, 6})];
            end
	    if(label == -1)
	        label = 2;
	    end
            %write to hdf5
            information(:, num_img) = [label roi]';

            num_img = num_img + 1;

            num_all = num_all + 1;
        end
    end
    if(num_img == chunksz + 1)
      fprintf('.');
      if(mod(num_all, 5000) == 0)
          fprintf('\n');
      end
      startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
      curr_dat_sz = store2hdf5(filename, src_data, information, ~created_flag, startloc, chunksz); 
      created_flag = true;
      totalct = curr_dat_sz(end);
      num_img = 1;
    end
end
fprintf('\n');

fclose(fid1);
fclose(fid2);
fclose(fid3);

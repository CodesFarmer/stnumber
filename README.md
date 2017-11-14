### CaffeTools		
*This project include some interface to help the users using caffe more convinent*		
  
#### Dependencies		
* [Caffe](http://caffe.berkeleyvision.org/)		
* [HDF5](https://support.hdfgroup.org/HDF5/)		
* [tinyXML2](http://www.grinninglizard.com/tinyxml2/)		
* [OpenCV](https://opencv.org/)		
  
#### Use Instruction  
After the depdendencies are setted up, you can compile and install our tools  
    #You are supposed under the root directory of our project  
    $mkdir build && cd build  
    $cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/where/you/want/install ..  
    $make install  
  
1.The program can generate bounding boxes with PNet, RNet, ONet of MTCNN  
  
To use the program, you should prepare your images and annotations in the follow struct  
  
---------------folder1-----------whateveryouwant--------------------cam0  
　　　　　　|　　　　　　　　　　　　　　　　　|------xml  
　　　　　　|  
　　　　　folder2-----------whateveryouwant--------------------cam0  
　　　　　　|　　　　　　　　　　　　　　　　　|------xml  
　　　　　　|  
　　　　　　.  
　　　　　　.  
　　　　　　.  
Then a filelists shuold be generated, which include the full path of the folder include the cam0 and xml, and the file name followed.  
For example, filelists.txt, like this  
/root/pathtofolder1/folder1/whateveryouwant/beforecam0 image\_0001  
/root/pathtofolder1/folder1/whateveryouwant/beforecam0 image\_0002  
...  
  
To generate patches, you can run the  
patchtest filelists.txt path/to/save/generated/samples  
You can refer examples in test/test\_patch.cc for more details  
  
2.The tools also support write the data into HDF5 database, you can refer the test/test\_hdf5.cc to create your HDF5 file  
  
3.You can extract the output from any layer of convolutional neural network(examples at test/test\_cnn.cc)  

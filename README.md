### CaffeTools		
*This project include some interface to help the users using caffe more convinent*		

The program can generate bounding boxes with PNet, RNet, ONet		

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
/root/pathtofolder1/folder1/whateveryouwant/beforecam0 image_0001		
/root/pathtofolder1/folder1/whateveryouwant/beforecam0 image_0002		
...

To generate patches, you can run the		
patchtest filelists.txt path/to/save/generated/samples

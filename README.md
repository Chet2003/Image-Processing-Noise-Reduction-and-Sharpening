# Image-Processing-Noise-Reduction-and-Sharpening
 A Python-based program that implements Image Sharpening (Unsharp Masking) and Noise Reduction (Average Filtering). The program uses the Nvidia Warp API to stimulate GPGPU programming by using highly efficient kernel operations.

The deliverable must be executed in an environment with Python installed. In 
addition, the warp, NumPy and Pillow libraries must be installed.  
• The files being provided for image processing can be greyscale, RGB. The program 
obtains the ability to process RGBA as an RGB file. 
• Files for processing must be in the same folder as the a3.py file.  
• Output files will be saved within the same folder as the a3.py file. 
• Command Line Arguments Format:  
o a3.py algType kernSize param inFileName outFileName 
▪ algType: Indicates the image processing algorithm. Utilize -s 
(sharpen) or -n (noise removal) 
▪ kernSize: Indicates the size of the kernel. Please ensure that the size 
is positive and odd.  
▪ Param: Indicates an additional parameters or value. For the sharpen 
algorithm, this value represents k for unsharp masking. For the noise 
removal algorithm, this value is not utilized but a dummy value will 
need to be passed (Ex:0).  
▪ inFileName: Name of the input file  
▪ outFileName: Name of the output file 
• Example of Command Line Arguments:  
o a3.py -n 3 0 noise_1.jpg noise_1NoiseReduct.jpg

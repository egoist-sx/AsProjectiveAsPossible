# APAP
C++ implementation of panorama stitching algorithm to deal with parallax as described in:
“As-projective-as-possible Image Stitching with Moving DLT”, Julio Zaragoza, Tat-Jun Chin, Michael Brown and David Suter, in Proc. Computer Vision and Pattern Recognition (CVPR), Portland, Oregon, USA, 2013.

and robust multiple model fitting (as optional alternative to RANSAC) as described in:
"Accelerated Hypothesis Generation for Multi-structure Robust Fitting", Tat-Jun Chin, Jin Yu, David Suter, European Conference on Computer Vision (ECCV), 2010.

This code has only been tested on Ubuntu 14.04, and requires following dependencies: GLEW, OpenGL, GLUT, DevIL, Eigen3. Various Sift detection and matching libraries and wrappers are provided including SiftGPU, VLFeat, OpenCV.



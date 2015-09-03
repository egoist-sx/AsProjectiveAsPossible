#pragma once

#include "sift.h"
#include <Eigen/Dense>

void detectSiftMatchWithVLFeat(const char* img1_path, const char* img2_path, Eigen::MatrixXf &match);
int detectSiftAndCalculateDescriptor(const char* img_path, double* &kp, vl_uint8* &descr);
int matchDescriptorWithRatioTest(const vl_uint8 *desc1, const vl_uint8 *desc2, int N1, int N2, int* &match);

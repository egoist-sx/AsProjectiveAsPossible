#pragma once

#include <vector>
#include <Eigen/Dense>
#include "SiftGPU.h"

void detectSiftMatchWithSiftGPU(const char* img1_path, const char* img2_path, Eigen::MatrixXf &match);
int detectFeatureSiftGPU(SiftGPU* sift, const char* path, std::vector<SiftGPU::SiftKeypoint> &keys, std::vector<float> &des);
SiftGPU* initSiftGPU();
int matchDescriptorSiftGPU(const std::vector<float> &des1, const std::vector<float> &des2, const std::vector<SiftGPU::SiftKeypoint> &keys1, const std::vector<SiftGPU::SiftKeypoint> &keys2, Eigen::MatrixXi &match);

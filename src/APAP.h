#pragma once

#include "Math.h"
#include "CVUtility.h"
#include "SiftGPUWrapper.h"
#include "VLFeatSiftWrapper.h"

void drawMatch(cv::Mat &img, const Eigen::MatrixXf &match, const Eigen::Matrix3f &inv_T1, const Eigen::Matrix3f &inv_T2);
void warpAndFuseImage(const cv::Mat &img1, const cv::Mat &img2, const Eigen::Matrix3f &H);
void warpAndFuseImageAPAP(const cv::Mat &img1, const cv::Mat &img2, const Eigen::MatrixXf &H, int offX, int offY, int cw, int ch, const Eigen::ArrayXf &X, const Eigen::ArrayXf &Y);
void APAP(const Eigen::MatrixXf &inlier, const Eigen::MatrixXf &A, const Eigen::Matrix3f &T1, const Eigen::Matrix3f &T2, int offX, int offY, int cw, int ch, const cv::Mat &img1, const cv::Mat &img2);
int GlobalHomography(const char* img1_path, const char* img2_path, Eigen::MatrixXf &inlier, Eigen::MatrixXf &A, Eigen::Matrix3f &T1, Eigen::Matrix3f &T2, int &offX, int &offY, int &cw, int &ch, cv::Mat &img1, cv::Mat &img2);

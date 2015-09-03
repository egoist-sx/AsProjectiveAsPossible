#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <random>
#include <vector>
#include <iostream>
#include <map>

extern float eps;

void noHomogeneous(Eigen::MatrixXf &mat);
Eigen::MatrixXf noHomogeneous(const Eigen::MatrixXf &mat);
void toHomogeneous(Eigen::MatrixXf &mat);
Eigen::MatrixXf toHomogeneous(const Eigen::MatrixXf &mat);
void normalizePts(Eigen::MatrixXf &mat, Eigen::Matrix3f &T);
void hnormalize(Eigen::MatrixXf &mat);
bool colinearity(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, const Eigen::Vector3f &p3);
void RandomSampling(int m, int N, std::vector<int> &sample);
void WeightedSampling(int m, int N, const Eigen::MatrixXi &resIndex, std::vector<int> &sample, int h);
void normalizeMatch(Eigen::MatrixXf &mat, Eigen::Matrix3f &T1, Eigen::Matrix3f &T2);
void fitHomography(Eigen::MatrixXf pts1, Eigen::MatrixXf pts2, Eigen::Matrix3f &H, Eigen::MatrixXf &A);
void noHomogeneous(Eigen::Vector3f &vec);
int maxOfSet(const std::vector<int> &data);
int minOfSet(const std::vector<int> &data);
float pdist2(float x1, float y1, float x2, float y2);
bool singleModelRANSAC(const Eigen::MatrixXf &data, int M, Eigen::MatrixXf &inlier);
bool multiModelRANSAC(const Eigen::MatrixXf &data, int M, Eigen::MatrixXf &inlier);
bool sampleValidTest(const Eigen::MatrixXf &pts1, const Eigen::MatrixXf &pts2);
void computeHomographyResidue(Eigen::MatrixXf pts1, Eigen::MatrixXf pts2, const Eigen::Matrix3f &H, Eigen::MatrixXf &residue);
Eigen::Matrix3f rollVector9f(const Eigen::VectorXf &h);
Eigen::VectorXf unrollMatrix3f(const Eigen::Matrix3f &H);
float intersect(const Eigen::VectorXi &permut1, const Eigen::VectorXi &permut2, int M, int h);
void computeIntersection(const Eigen::MatrixXi &thisIndex, const Eigen::MatrixXi& index, int h, Eigen::ArrayXf &new_w);
void sortResidueForIndex(const Eigen::MatrixXf &resMat, int colLow, int colHigh, Eigen::MatrixXi &resIndex);
void filterPointAtInfinity(Eigen::MatrixXf &pts1, Eigen::MatrixXf &pts2);

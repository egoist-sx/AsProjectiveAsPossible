#include "Math.h"
#include <float.h>

#ifndef THRESHOLD
  #define THRESHOLD 0.1
#endif

using namespace std;
using namespace Eigen;

void normalizePts(MatrixXf &mat, Matrix3f &T) {

  float cx = mat.col(0).mean();
  float cy = mat.col(1).mean();
  mat.array().col(0) -= cx;
  mat.array().col(1) -= cy;
  
  float sqrt_2 = sqrt(2);
  float meandist = (mat.array().col(0)*mat.array().col(0) + mat.array().col(1)*mat.array().col(1)).sqrt().mean();
  float scale = sqrt_2/meandist;
  mat.leftCols<2>().array() *= scale; 

  T << scale, 0, -scale*cx, 0, scale, -scale*cy, 0, 0, 1;
}

void toHomogeneous(MatrixXf &mat) {
  MatrixXf temp;
  if (mat.cols() == 2) {
    temp.resize(mat.rows(), 3);
    temp.leftCols<2>() = mat.leftCols<2>();
    temp.col(2).setConstant(1);
    mat = temp;
  } else if (mat.cols() == 4) {
    temp.resize(mat.rows(), 6);
    temp.leftCols<2>() = mat.leftCols<2>();
    temp.col(2).setConstant(1);
    temp.block(0, 3, mat.rows(), 2) = temp.block(0, 2, mat.rows(), 2);
    temp.col(5).setConstant(1);
    mat = temp;
  } else 
    cout << "toHomogeneous with wrong dimension" << endl;
}

void noHomogeneous(MatrixXf &mat) {
  MatrixXf temp;
  if (mat.cols() == 3) {
    temp.resize(mat.rows(), 2);
    temp.col(0).array() = mat.col(0).array()/mat.col(2).array();
    temp.col(1).array() = mat.col(1).array()/mat.col(2).array();
    mat = temp;
  } else 
    cout << "toHomogeneous with wrong dimension" << endl;
}

void noHomogeneous(Vector3f &vec) {
  if (abs(vec[2]) < FLT_EPSILON)
    cerr << "Divide by 0" << endl;
  vec[0] = vec[0]/vec[2];
  vec[1] = vec[1]/vec[2];
  vec[2] = 1;
}

// normalizeMatch respect to "In defense of eight point algorithm"
void normalizeMatch(MatrixXf &mat, Matrix3f &T1, Matrix3f &T2) {
  MatrixXf pts1 = mat.leftCols<3>();
  MatrixXf pts2 = mat.block(0, 3, mat.rows(), 3);
  normalizePts(pts1, T1);
  normalizePts(pts2, T2);
  mat.leftCols<3>() = pts1;
  mat.block(0, 3, mat.rows(), 3) = pts2;
}

// Test if three point are colinear
bool colinearity(const Vector3f &p1, const Vector3f &p2, const Vector3f &p3) {
  if (abs(p1.dot(p2.cross(p3))) < FLT_EPSILON)
    return true;
  else
    return false;
}

void RandomSampling(int m, int N, vector<int> &samples) {
  samples.reserve(m);
  random_device rd;
  mt19937 randomGenerator(rd());
  // too slow
  vector<int> numberBag(N);
  for (int i = 0; i < N; i++) 
    numberBag[i] = i;

  int max = static_cast<int>(numberBag.size()-1);
  for (int i = 0; i < m; i++) {
    uniform_int_distribution<> uniformDistribution(0, max);
    int index = uniformDistribution(randomGenerator);
    swap(numberBag[index], numberBag[max]);
    samples[N-1-max] = numberBag[max];
    max--;
  }
}

void fitHomography(MatrixXf pts1, MatrixXf pts2, Matrix3f &H, MatrixXf &A) {
  int psize = pts1.rows();
  A.resize(psize*2, 9);
  for (auto i = 0; i < psize; i++) {
    Vector3f p1 = pts1.row(i);
    Vector3f p2 = pts2.row(i);
    A.row(i*2) << 0, 0, 0, -p1[0], -p1[1], -p1[2], p2[1]*p1[0], p2[1]*p1[1], p2[1]*p1[2];
    A.row(i*2+1) << p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2];
  }
  
  JacobiSVD<MatrixXf, HouseholderQRPreconditioner> svd(A, ComputeFullV);
  MatrixXf V = svd.matrixV();
  VectorXf h = V.col(V.cols()-1);
  H = rollVector9f(h);
}

int maxOfSet(const vector<int> &data) {
  int max = data[0];
  for (int i = 1; i < data.size(); i++)
    if (data[i] > max)
      max = data[i];

  return max;
}

int minOfSet(const vector<int> &data) {
  int min = data[0];
  for (int i = 1; i < data.size(); i++)
    if (data[i] < min)
      min = data[i];
  return min;
}

float pdist2(float x1, float y1, float x2, float y2) {
  return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

bool singleModelRANSAC(const MatrixXf &data, int M, MatrixXf &inlier) {
  int maxdegen = 10;
  int dataSize = data.rows();
  int psize = 4;
  MatrixXf x1 = data.block(0, 0, data.rows(), 3);
  MatrixXf x2 = data.block(0, 3, data.rows(), 3);
  vector<int> sample;
  MatrixXf pts1(4, 3);
  MatrixXf pts2(4, 3);
  int maxInlier = -1;
  MatrixXf bestResidue;
  for (int m = 0; m < M; m++) {
    int degencount = 0;
    int isdegen = 1;
    while (isdegen==1 && degencount < maxdegen) {
      degencount ++;
      RandomSampling(psize, dataSize, sample);
      for (int i = 0; i < psize; i++) {
        pts1.row(i) = x1.row(sample[i]);
        pts2.row(i) = x2.row(sample[i]);
      }
      if (sampleValidTest(pts1, pts2))
          isdegen = 0;
    }
    if (isdegen) {
      cout << "Cannot find valid p-subset" << endl;
      return false;
    }
    Matrix3f local_H;
    MatrixXf local_A;
    fitHomography(pts1, pts2, local_H, local_A);

    MatrixXf residue;
    computeHomographyResidue(x1, x2, local_H, residue);
    int inlierCount = (residue.array() < THRESHOLD).count();
    if (inlierCount > maxInlier) {
      maxInlier = inlierCount;
      bestResidue = residue;
    }
  }
  inlier.resize(maxInlier, data.cols());
  int transferCounter = 0;
  for (int i = 0; i < dataSize; i++) {
    if (bestResidue(i) < THRESHOLD) {
      inlier.row(transferCounter) = data.row(i);
      transferCounter++;
    }
  }
  if (transferCounter != maxInlier) {
    cout << "RANSAC result size does not match!!!!" << endl;
    return false;
  }
  return true;
}

void WeightedSampling(int m, int N, const MatrixXi &resIndex, vector<int> &sample, int h) {
  sample.reserve(m);
  vector<int> seedIndex;
  RandomSampling(1, N, seedIndex);
  sample[0] = seedIndex[0];
  int prevSelected = seedIndex[0];
  ArrayXf w(N);
  w.setConstant(1);
  for (int i = 1; i < m; i++) {
    ArrayXf new_w(N);
    computeIntersection(resIndex.row(prevSelected), resIndex, h, new_w);
    w(prevSelected) = 0;
    w *= new_w;
    if (w.sum() > 0) {
      map<double, int> cumulative;
      typedef std::map<double, int>::iterator it;
      double acc = 0;
      for (int j = 0; j < N; j++) {
        acc += w(j);
        cumulative[acc] = j; 
      }
      double linear = rand()*acc/RAND_MAX;
      prevSelected = cumulative.upper_bound(linear)->second;
      sample[i] = prevSelected;
    } else {
      vector<int> temp_sample;
      RandomSampling(1, N, temp_sample);
      prevSelected = temp_sample[0];
      sample[i] = prevSelected;
    }
  }
}

void computeIntersection(const MatrixXi &thisIndex, const MatrixXi &index, int h, ArrayXf &new_w) {
  int dataSize = index.rows();
  int M = index.cols();
  for (int i = 0 ; i < dataSize; i++) 
    new_w(i) = intersect(thisIndex.transpose(), index.row(i).transpose(), M, h);
}

float intersect(const VectorXi &permut1, const VectorXi &permut2, int M, int h) {
  int *symbolTable = (int*) malloc(sizeof(int)*M);
  memset(symbolTable, 0, sizeof(int)*M);
  for (int i = 0; i < h; i++)
    symbolTable[permut1(i)] = 1;
  int acc = 0;
  for (int i = 0; i < h; i++)
    if (symbolTable[permut2(i)] == 1)
      acc++;
  free(symbolTable);
  return (acc*1.f)/h;
}

void sortResidueForIndex(const MatrixXf &resMat, int colLow, int colHigh, MatrixXi &resIndex) {
  struct ResIndexPair {
    float res;
    size_t idx;
    ResIndexPair(float _res, size_t _idx):res(_res), idx(_idx){}
  };
  vector<ResIndexPair> data;
  int rows = resMat.rows();
  int cols = resMat.cols();
  // sort each model in range
  for (int i = colLow; i < colHigh; i++) {
    for (int j = 0; j < rows; j++) 
      data.push_back(ResIndexPair(resMat(j, i), j));
    sort(data.begin(), data.end(), [](const ResIndexPair &a, const ResIndexPair &b) {return a.res < b.res;});
    for (int j = 0; j < rows; j++) 
      resIndex(j, i) = data[j].idx;
    vector<ResIndexPair>().swap(data);
  }
}

bool multiModelRANSAC(const MatrixXf &data, int M, MatrixXf &inlier) {
  int maxdegen = 10;
  int dataSize = data.rows();
  int psize = 4;
  int blockSize = 10;
  MatrixXf x1 = data.block(0, 0, data.rows(), 3);
  MatrixXf x2 = data.block(0, 3, data.rows(), 3);
  vector<int> sample;
  MatrixXf pts1(4, 3);
  MatrixXf pts2(4, 3);

  int h = 0;
  MatrixXf Hs(M, 9);
  MatrixXf inx(M, psize);
  MatrixXf res(dataSize, M);
  MatrixXi resIndex(dataSize, M);

  for (int m = 0; m < M; m++) {
    int degencount = 0;
    int isdegen = 1;
    
    while (isdegen==1 && degencount < maxdegen) {
      degencount++;
      if (m < blockSize)
        RandomSampling(psize, dataSize, sample);
      else 
        WeightedSampling(psize, dataSize, resIndex, sample, h);
      for (int i = 0; i < psize; i++) {
        pts1.row(i) = x1.row(sample[i]);
        pts2.row(i) = x2.row(sample[i]);
      }
      if (sampleValidTest(pts1, pts2))
        isdegen = 0;
    }
    if (isdegen) {
      cout << "Cannot find valid p-subset" << endl;
      return false;
    }
    for (int i = 0; i < psize; i++)
      inx(m, i) = sample[i]; 

    Matrix3f temp_H;
    MatrixXf temp_A, localResidue;
    fitHomography(pts1, pts2, temp_H, temp_A);
    computeHomographyResidue(x1, x2, temp_H, localResidue);
    Hs.row(m) = unrollMatrix3f(temp_H);
    res.col(m) = localResidue;
    if (m >= (blockSize-1) && (m+1)%blockSize == 0) {
      h = round(0.1f*m);
      sortResidueForIndex(res, (m/blockSize)*blockSize, ((m+1)/blockSize)*blockSize, resIndex);
    }
  }

  VectorXf bestModel(M);
  bestModel.setZero();
  int bestIndex = 0;
  int bestCount = -1;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < dataSize; j++) 
      if (res(j, i) < THRESHOLD)
        bestModel(i) += 1;
    if (bestModel(i) > bestCount) {
      bestIndex = i;
      bestCount = bestModel(i);
    }
  }

  VectorXf bestModelRes = res.col(bestIndex);
  int inlierCount = (bestModelRes.array() < THRESHOLD).count();
  inlier.resize(inlierCount, data.cols());
  int runningIdx = 0;
  for (int i = 0; i < dataSize; i++) 
    if (bestModelRes(i) < THRESHOLD) {
      inlier.row(runningIdx) = data.row(i);
      runningIdx ++;
    }

  return true;
}

// Sample degeneration test, return false if at least three point are colinear
bool sampleValidTest(const MatrixXf &pts1, const MatrixXf &pts2) {
  return !(colinearity(pts1.row(1), pts1.row(2), pts1.row(3)) ||
           colinearity(pts1.row(0), pts1.row(1), pts1.row(2)) ||
           colinearity(pts1.row(0), pts1.row(2), pts1.row(3)) ||
           colinearity(pts1.row(0), pts1.row(1), pts1.row(3)) ||
           colinearity(pts2.row(1), pts2.row(2), pts2.row(3)) ||
           colinearity(pts2.row(0), pts2.row(1), pts2.row(2)) ||
           colinearity(pts2.row(0), pts2.row(2), pts2.row(3)) ||
           colinearity(pts2.row(0), pts2.row(1), pts2.row(3)));
}

void filterPointAtInfinity(MatrixXf &pts1, MatrixXf &pts2) {
  int finiteCount = 0;
  for (int i = 0; i < pts1.rows(); i++) {
    if (abs(pts1(i, 2)) > FLT_EPSILON && abs(pts2(i, 2) > FLT_EPSILON)) 
      finiteCount++;
  }
  MatrixXf temp_pts1, temp_pts2;
  temp_pts1.resize(finiteCount, pts1.cols());
  temp_pts2.resize(finiteCount, pts2.cols());
  int idx = 0;
  for (int i = 0; i < pts1.rows(); i++) {
    if (abs(pts1(i, 2)) > FLT_EPSILON && abs(pts2(i, 2) > FLT_EPSILON)) {
      temp_pts1.row(idx) = pts1.row(i); 
      temp_pts2.row(idx) = pts2.row(i); 
      idx++;
    }
  }
  pts1 = temp_pts1;
  pts2 = temp_pts2;
}

void computeHomographyResidue(MatrixXf pts1, MatrixXf pts2, const Matrix3f &H, MatrixXf &residue){
  // cross residue
  filterPointAtInfinity(pts1, pts2);
  residue.resize(pts1.rows(), 1);
  MatrixXf Hx1 = (H*pts1.transpose()).transpose();
  MatrixXf invHx2 = (H.inverse()*pts2.transpose()).transpose();

  noHomogeneous(Hx1);
  noHomogeneous(invHx2);
  noHomogeneous(pts1);
  noHomogeneous(pts2);

  MatrixXf diffHx1pts2 = Hx1 - pts2;
  MatrixXf diffinvHx2pts1 = invHx2 - pts1;
  residue = diffHx1pts2.rowwise().squaredNorm() + diffinvHx2pts1.rowwise().squaredNorm();
}

Matrix3f rollVector9f(const VectorXf &h) {

  Matrix3f H;
  H << h[0], h[1], h[2],
       h[3], h[4], h[5],
       h[6], h[7], h[8];
  return H;
}

VectorXf unrollMatrix3f(const Matrix3f &H) {

  VectorXf h(9);
  h << H(0, 0), H(0, 1), H(0, 2),
       H(1, 0), H(1, 1), H(1, 2),
       H(2, 0), H(2, 1), H(2, 2);
  return h;
}

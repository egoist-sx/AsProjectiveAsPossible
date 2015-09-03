#include "SiftGPUWrapper.h"
#include <iostream>

using namespace std;
using namespace Eigen;

int detectFeatureSiftGPU(SiftGPU* sift, const char* path, vector<SiftGPU::SiftKeypoint> &keys, vector<float> &des) {
  if (sift->RunSIFT(path))
  {
    int num;
    num = sift->GetFeatureNum();
    keys.resize(num);
    des.resize(num*128);
    sift->GetFeatureVector(&keys[0], &des[0]);
    return 1;
  }
  return 0;
}

SiftGPU* initSiftGPU() {
  SiftGPU* sift = new SiftGPU;
  char *argv[] = {(char *)"-fo", (char *)"-1", (char *)"-v", (char *)"1"};
  sift->ParseParam(4, argv);
  int support = sift->CreateContextGL();
  if (support != SiftGPU::SIFTGPU_FULL_SUPPORTED) 
    return NULL;
  else 
    return sift;
}

// Rows of match is in format
// x1 y1 x2 y2
int matchDescriptorSiftGPU(const vector<float> &des1, const vector<float> &des2, const vector<SiftGPU::SiftKeypoint> &keys1, const vector<SiftGPU::SiftKeypoint> &keys2, MatrixXf &match) {
  SiftMatchGPU *matcher = new SiftMatchGPU(4096);
  int match_buffer[4096][2];
  if (matcher->VerifyContextGL() == 0) {
    cout << "Fail to initialize SiftMatchGPU" << endl;
    return -1;
  }

  matcher->SetDescriptors(0, des1.size()/128, &des1[0]);
  matcher->SetDescriptors(1, des2.size()/128, &des2[0]);

  // Guided SiftGPU match
  // 1/4 size
  //float H[3][3] = {0.552205, 0.36639, -190.011, -0.000139628, 0.740511, -252.265, 0, 0.000723292, 0.174118};
  //float H[3][3] = {0.552205, -0.000139628, 0, 0.36639, 0.740511, 0.000723292, -190.011, -252.265, 0.174118};
  // full size
  //float H[3][3] = {0.557203, 0.353675, -739.85, 0.000503741, 0.734669, -996.908, 0, 0.000175132, 0.184016};
  //float H[3][3] = {0.557203, 0.000503741, 0, 0.353675, 0.734669, 0.000175132, -739.85, -996.908, 0.184016};
  //int num_match = matcher->GetGuidedSiftMatch(4096, match_buffer, H, NULL, 4, 4, 50, 30, 0);
  
  // dist threshold only, remove ratio test
  int num_match = matcher->GetSiftMatch(4096, match_buffer, 0.7, 0.9, 1);
  //int num_match = matcher->GetSiftMatch(4096, match_buffer);
  cout << "num match " << num_match << endl;

  match.resize(num_match, 6);
  for (int i = 0; i < num_match; i++) 
    match.row(i) << keys1[match_buffer[i][0]].x, keys1[match_buffer[i][0]].y, 1, keys2[match_buffer[i][1]].x, keys2[match_buffer[i][1]].y, 1;

  return num_match;
}

void detectSiftMatchWithSiftGPU(const char* img1_path, const char* img2_path, MatrixXf &match) {

  SiftGPU *sift = initSiftGPU();
  if (sift == NULL) {
    cout << "Fail to initialize SiftGPU" << endl;
    return;
  }
    
  vector<float> des1(1), des2(1);
  vector<SiftGPU::SiftKeypoint> keys1(1), keys2(1);

  if (detectFeatureSiftGPU(sift, img1_path, keys1, des1) != 1) {
    cout << "Fail to detect feature in first image" << endl;
    return;
  }

  if (detectFeatureSiftGPU(sift, img2_path, keys2, des2) != 1) {
    cout << "Fail to detect feature in second image" << endl;
    return;
  }

  delete sift;

  int num_match = matchDescriptorSiftGPU(des1, des2, keys1, keys2, match);
  if (num_match > 0) 
    cout << "Number of match found: " << num_match << endl;
  else
    return;
}

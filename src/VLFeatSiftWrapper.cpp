#include "VLFeatSiftWrapper.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits.h>

using namespace Eigen;
using namespace cv;
using namespace std;

void detectSiftMatchWithVLFeat(const char* img1_path, const char* img2_path, Eigen::MatrixXf &match) {

  int *m = 0;
  double *kp1 = 0, *kp2 = 0;
  vl_uint8 *desc1 = 0, *desc2 = 0;

  int nkp1 = detectSiftAndCalculateDescriptor(img1_path, kp1, desc1);
  int nkp2 = detectSiftAndCalculateDescriptor(img2_path, kp2, desc2);
  cout << "num kp1: " << nkp1 << endl;
  cout << "num kp2: " << nkp2 << endl;
  int nmatch = matchDescriptorWithRatioTest(desc1, desc2, nkp1, nkp2, m);
  cout << "num match: " << nmatch << endl;
  match.resize(nmatch, 6);
  for (int i = 0; i < nmatch; i++) {
    int index1 = m[i*2+0];
    int index2 = m[i*2+1];
    match.row(i) << kp1[index1*4+1], kp1[index1*4+0], 1, kp2[index2*4+1], kp2[index2*4+0], 1;
  }

  free(kp1);
  free(kp2);
  free(desc1);
  free(desc2);
  free(m);
}

int detectSiftAndCalculateDescriptor(const char* img_path, double* &kp, vl_uint8* &descr) {

  Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
  int height = img.size[0];
  int width = img.size[1];
  float *data = (float*) malloc(height*width*sizeof(float));
  uchar *rawData = img.data;
  for(int i = 0; i < height; i++) 
    for(int j = 0; j < width; j++) 
      data[i*width+j] = (float)rawData[i*width+j];
  int M = height;
  int N = width;
  const vl_sift_pix* vlData = (vl_sift_pix*) data;

  // Octaves
  int O = -1;
  // Levels
  int S = 3;
  int o_min = 0;

  double edge_threshold = 500;
  double peak_threshold = 0;

  VlSiftFilt *filt;
  vl_bool first;
  int reserved = 0, nkp = 0, i, j, q;

  filt = vl_sift_new(N, M, O, S, o_min);
  vl_sift_set_peak_thresh(filt, peak_threshold);
  vl_sift_set_edge_thresh(filt, edge_threshold);

  first = 1;
  while (1) {
    int err;
    const VlSiftKeypoint* keys = 0;
    int nkeys = 0;

    if (first) {
      err = vl_sift_process_first_octave(filt, vlData);
      first = 0;
    } else {
      err = vl_sift_process_next_octave(filt);
    }

    if (err) {
      break;
    }

    vl_sift_detect(filt);
    keys = vl_sift_get_keypoints(filt);
    nkeys = vl_sift_get_nkeypoints(filt);
    i = 0;

    for (; i < nkeys; i++) {
      double angles[4];
      int nangles;
      const VlSiftKeypoint* k;

      k = keys + i;
      nangles = vl_sift_calc_keypoint_orientations(filt, angles, k);

      for (q = 0; q < nangles; ++q) {

        vl_sift_pix buf[128];
        vl_sift_calc_keypoint_descriptor(filt, buf, k, angles[q]);
        if (reserved < nkp+1) {
          reserved += 2*nkeys;
          kp = (double*) realloc(kp, 4*sizeof(double)*reserved);
          descr = (vl_uint8*) realloc(descr, 128*sizeof(vl_uint8)*reserved);
        }
        kp[4*nkp+0] = k->y+1;
        kp[4*nkp+1] = k->x+1;
        kp[4*nkp+2] = k->sigma;
        kp[4*nkp+3] = M_PI/2-angles[q];
        for (j = 0; j < 128; j++) {
          float x = 512.f*buf[j];
          x = (x < 255.f)?x:255.f;
          descr[128*nkp+j] = (vl_uint8) x;
        }
        ++nkp;
      }
    }
  }
  free(data);
  return nkp;
}

int matchDescriptorWithRatioTest(const vl_uint8 *desc1, const vl_uint8 *desc2, int N1, int N2, int* &match) {
  int ND = 128;
  float thresh = 1.5;
  int matchCount = 0;
  int *tempMatch = (int*) malloc(sizeof(int)*max(N1, N2)*2);
  int reserved = max(N1,N2);

  for (int k1 = 0; k1 < N1; k1++) {
    long best = LONG_MAX;
    long second_best = LONG_MAX;
    int bestk = -1;

    for (int k2 = 0; k2 < N2; k2++) {
      long acc = 0;
      for (int bin = 0; bin < ND; bin++) {
        long delta = (long)desc1[k1*ND+bin] - (long)desc2[k2*ND+bin];
        acc += delta*delta;
        if (acc >= second_best)
          break;
      }

      if (acc < best) {
        second_best = best;
        best = acc;
        bestk = k2;
      } else if (acc < second_best) 
        second_best = acc;
    }

    if (reserved < matchCount + 1) {
      reserved += max(N1, N2);
      tempMatch = (int*) realloc(tempMatch, sizeof(int)*2*reserved);
    }

    if (thresh*(float)best < (float)second_best && bestk != -1) {
      tempMatch[matchCount*2+0] = k1;
      tempMatch[matchCount*2+1] = bestk;
      matchCount++;
    } 
  }

  match = (int*) malloc(sizeof(int)*2*matchCount);
  for (int i = 0; i < matchCount; i++) {
    match[i*2+0] = tempMatch[i*2+0];
    match[i*2+1] = tempMatch[i*2+1];
  }

  free(tempMatch);
  return matchCount;
}

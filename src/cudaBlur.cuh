#include <string>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;

const int KERNEL_SIZE = 21;
const int BLOCK_SIZE = 32;

__global__ void blurKernel(uchar *d_img, float *d_blur, int H_in, int W_in, int H_out, int W_out);

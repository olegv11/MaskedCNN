#include <assert.h>
#include "Cuda.hpp"
#include <stdio.h>

extern "C" {

__global__ void ReLuKernel(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x)
    {
        y[i] = (x[i] > 0.0) ? x[i] : 0.0;
        delta[i] = (x[i] > 0.0) ? 1.0 : 0.0;
    }
}

__global__ void IdKernel(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x)
    {
        y[i] = x[i];
        delta[i] = 1;
    }
}

__global__ void SigmoidKernel(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x)
    {
        y[i] = 1.0 / (1.0 + expf(-x[i]));
        delta[i] = y[i] * (1 - y[i]);
    }
}

__global__ void TanhKernel(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x)
    {
        float t = expf(2 * x[i]);
        y[i] = (t - 1) / (t + 1);
        delta[i] = (1 - y[i] * y[i]);
    }
}


void ReLu_activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    ReLuKernel<<<GET_BLOCKS(num), NUM_THREADS>>>(x,y,delta,num);
}

void Id_activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    IdKernel<<<GET_BLOCKS(num), NUM_THREADS>>>(x,y,delta,num);
}

void Sigmoid_activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    SigmoidKernel<<<GET_BLOCKS(num), NUM_THREADS>>>(x,y,delta,num);
}

void Tanh_activate_gpu(const float *__restrict__ x, float *__restrict__ y, float *__restrict__ delta, int num)
{
    TanhKernel<<<GET_BLOCKS(num), NUM_THREADS>>>(x,y,delta,num);
}

__global__ void im2col_gpu_kernel(const int n, const float* dataIm,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* dataCol)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x)
    {
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;
        float* data_col_ptr = dataCol;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const float* data_im_ptr = dataIm;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                int h_im = h_offset + i;
                int w_im = w_offset + j;
                *data_col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                    data_im_ptr[i * width + j] : 0;
                //if (index == 7)
                //printf("data_col_ptr = %f\n", (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                //           data_im_ptr[i * width + j] : 0);
                data_col_ptr += height_col * width_col;
            }
        }
    }
}



void im2col_gpu(const float *dataIm, int inputChannels, int inputHeight, int inputWidth,
                int filterSize, int pad, int stride, float *dataCol)
{
    const int outputHeight = (inputHeight + 2 * pad - filterSize) / stride + 1;
    const int outputWidth = (inputWidth + 2 * pad - filterSize) / stride + 1;
    int numKernels = inputChannels * outputHeight * outputWidth;
    //printf("HEY HEY HEY %d %d %d %d %d %d %d %d\n", numKernels, inputHeight, inputWidth, filterSize, pad, stride, outputHeight, outputWidth);
    im2col_gpu_kernel<<<(inputChannels + 512 - 1) / 512, 512>>>(numKernels, dataIm, inputHeight, inputWidth, filterSize, filterSize, pad,
                       pad, stride, stride, outputHeight, outputWidth, dataCol);
}

void hello_world()
{
    printf("ahahaha!!!\n");
}

}

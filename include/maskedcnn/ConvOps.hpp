#pragma once
#include "Tensor.hpp"
#include "Util.hpp"

namespace MaskedCNN {



int rot180(int f, int filterSize)
{
    return filterSize - f - 1;
}

void convolution(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float>& out, int filterSize, int stride, int pad)
{
    const int outputChannels = out.dimensions()[0];
    const int outputHeight = out.dimensions()[1];
    const int outputWidth = out.dimensions()[2];
    const int inputChannels = input.dimensions()[0];
    const int inputHeight = input.dimensions()[1];
    const int inputWidth = input.dimensions()[2];

    out.zero();

    for (int ay = 0, y = -pad; ay < outputHeight; ay++, y += stride)
    {
        for (int ax = 0, x = -pad; ax < outputWidth; ax++, x += stride)
        {
            for (int fy = 0; fy < filterSize; fy++)
            {
                for (int fx = 0; fx < filterSize; fx++)
                {
                    int oy = y + fy;
                    int ox = x + fx;

                    if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                    {
                        for (int d = 0; d < outputChannels; d++)
                        {
                            for (int fd = 0; fd < inputChannels; fd++)
                            {
                                //std::cout << "out(" << d << "," << ay << "," << ax << ") "
                                //          << out(d, ay, ax) << "+=" << input(fd, oy, ox) << "*" << filter(d, fd, fy, fx) << std::endl;
                                out(d, ay, ax) += input(fd, oy, ox) * filter(d, fd, fy, fx);
                            }
                        }
                    }
                }
            }
        }
    }
}


void transposedConvolution(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float>& out, int filterSize, int stride, int pad)
{
    // Doing transposed convolution here
    // https://arxiv.org/pdf/1603.07285.pdf

    const int outputChannels = out.dimensions()[0];
    const int outputHeight = out.dimensions()[1];
    const int outputWidth = out.dimensions()[2];
    const int inputChannels = input.dimensions()[0];
    const int inputHeight = input.dimensions()[1];
    const int inputWidth = input.dimensions()[2];

    // "inserting" (stride-1) virtual zeroes in the transposed input between each "real" element
    const int transposedHeight = inputHeight + (inputHeight - 1) * (stride - 1);
    const int transposedWidth = inputWidth + (inputWidth - 1) * (stride - 1);

    const int virtualPad = filterSize - pad - 1;
    assert(virtualPad >= 0); // TODO: correctly compute it in cases where padding is more than full padding

    // Additional padding from left and up
    const int padLeft = (outputWidth + 2 * pad - filterSize) % stride;
    const int padUp = (outputHeight + 2 * pad - filterSize) % stride;

    const int shiftLeft = -(virtualPad + padLeft);
    const int shiftUp = -(virtualPad + padUp);

    out.zero();

    for (int ay = shiftUp; ay < transposedHeight + virtualPad - filterSize + 1; ay++)
    {
        for (int ax = shiftLeft; ax < transposedWidth + virtualPad - filterSize + 1; ax++)
        {
            for (int fy = 0; fy < filterSize; fy++)
            {
                int oy = ay + rot180(fy, filterSize);
                if ((oy % stride != 0) || oy < 0 || oy / stride >= inputHeight) continue;

                for (int fx = 0; fx < filterSize; fx++)
                {
                    int ox = ax + rot180(fx, filterSize);
                    if ((ox % stride != 0) || ox < 0 || ox / stride >= inputWidth) continue;

                    for (int d = 0; d < outputChannels; d++)
                    {
                        for (int fd = 0; fd < inputChannels; fd++)
                        {
                            /*std::cout << "out(" << d << "," << (ay - shiftUp)/ stride << "," << (ax  - shiftLeft)/ stride <<") "
                                      << out(d, (ay - shiftUp)/ stride, (ax  - shiftLeft)/ stride) << "+=input(" << fd <<","<<oy / stride<<","<<ox/stride<<") * filter("
                                      <<fd<<","<<d<<","<<fy<<","<<fx<<") "
                                      << input(fd, oy / stride, ox / stride) << "*" << filter(fd, d, fy, fx) << std::endl;*/
                            out(d, (ay - shiftUp), (ax  - shiftLeft)) += input(fd, oy / stride, ox / stride) * filter(fd, d, fy, fx);
                        }
                    }
                }
            }
        }
    }
    //std::cout << std::endl;
}

// Thanks to https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp for the reference implementation

void im2col(const Tensor<float>& im, int inputChannels, int inputHeight, int inputWidth, int filterSize, int pad, int stride, Tensor<float>& col)
{
    auto dataCol = col.dataAddress();
    auto dataIm = im.dataAddress();
    const int outputHeight = (inputHeight + 2 * pad - filterSize) / stride + 1;
    const int outputWidth = (inputWidth + 2 * pad - filterSize) / stride + 1;
    const int channelSize = inputHeight * inputWidth;

    for (int channel = inputChannels; channel--; dataIm += channelSize)
    {
        for (int fy = 0; fy < filterSize; fy++)
        {
            for (int fx = 0; fx < filterSize; fx++)
            {
                int y = -pad + fy;
                for (int outputRows = outputHeight; outputRows; outputRows--)
                {
                    if (y < 0 || y >= inputHeight)
                    {
                        for (int outputCols = outputWidth; outputCols; outputCols--)
                        {
                            *(dataCol++) = 0;
                        }
                    }
                    else
                    {
                        int x = -pad + fx;
                        for (int outputCols = outputWidth; outputCols; outputCols--)
                        {
                            if (x >= 0 && x < inputWidth)
                            {
                                *(dataCol++) = dataIm[y * inputWidth + x];
                            }
                            else
                            {
                                *(dataCol++) = 0;
                            }
                            x += stride;
                        }
                    }
                    y += stride;
                }
            }
        }
    }
}

void col2im(const Tensor<float>& col, int filterSize, int pad, int stride, Tensor<float>& data)
{
    data.zero();

    auto dataCol = col.dataAddress();
    auto dataIm = data.dataAddress();
    const int inputHeight = data.dimensions()[1];
    const int inputWidth = data.dimensions()[2];
    const int outputHeight = (inputHeight + 2 * pad - filterSize) / stride + 1;
    const int outputWidth = (inputWidth + 2 * pad - filterSize) / stride + 1;
    const int channelSize = inputHeight * inputWidth;

    for (int channel = channelSize; channel--; dataIm += channelSize)
    {
        for (int fy = 0; fy < filterSize; fy++)
        {
            for (int fx = 0; fx < filterSize; fx++)
            {
                int y = -pad;
                for (int outputRows = outputHeight; outputRows; outputRows--)
                {
                    if (y < 0 || y >= inputHeight)
                    {
                        dataCol += outputWidth;
                    }
                    else
                    {
                        int x = -pad;
                        for (int outputCols = outputWidth; outputCols; outputCols--)
                        {
                            if (x >= 0 && x < inputWidth)
                            {
                                dataIm[y * inputWidth + x] += *dataCol;
                            }
                            dataCol++;
                            x += stride;
                        }
                    }
                    y += stride;
                }
            }
        }
    }
}

void convolutionIm2Col(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float> &colBuffer, Tensor<float>& out, int filterSize, int stride, int pad)
{
    const int outputChannels = out.dimensions()[0];
    const int outputHeight = out.dimensions()[1];
    const int outputWidth = out.dimensions()[2];
    const int inputChannels = input.dimensions()[0];
    const int inputHeight = input.dimensions()[1];
    const int inputWidth = input.dimensions()[2];

    colBuffer.resize(std::vector<int>{inputChannels*filterSize*filterSize, outputHeight * outputWidth});

    im2col(input, inputChannels, inputHeight, inputWidth, filterSize, pad, stride, colBuffer);

    int m = outputChannels;
    int n = outputHeight * outputWidth;
    int k = inputChannels * filterSize * filterSize;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.0, filter.dataAddress(), k, colBuffer.dataAddress(),
                n, 0., out.dataAddress(), n);
}


}

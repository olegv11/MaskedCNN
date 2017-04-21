#pragma once
#include "Tensor.hpp"

namespace MaskedCNN {



int rot180(int f, int filterSize)
{
    return filterSize - f - 1;
}

void convolution(const Tensor<float> &input, const Tensor<float> &filter, Tensor<float> &out, int filterSize, int stride, int pad)
{
    int outputChannels = out.dimensions()[0];
    int outputHeight = out.dimensions()[1];
    int outputWidth = out.dimensions()[2];
    int filterDepth = input.dimensions()[0];
    int inputHeight = input.dimensions()[1];
    int inputWidth = input.dimensions()[2];

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
                            for (int fd = 0; fd < filterDepth; fd++)
                            {
                                out(d, ay, ax) += input(fd, oy, ox) * filter(d, fd, fy, fx);
                            }
                        }
                    }
                }
            }
        }
    }
}


void transposedConvolution(const Tensor<float>& input, const Tensor<float>& filter, Tensor<float> &out, int filterSize, int stride, int pad)
{
    // Doing transposed convolution here
    // https://arxiv.org/pdf/1603.07285.pdf

    int outputChannels = out.dimensions()[0];
    int outputHeight = out.dimensions()[1];
    int outputWidth = out.dimensions()[2];
    int inputChannels = input.dimensions()[0];
    int inputHeight = input.dimensions()[1];
    int inputWidth = input.dimensions()[2];

    // "inserting" (stride-1) virtual zeroes in the transposed input between each "real" element
    int transposedHeight = inputHeight + (inputHeight - 1) * (stride - 1);
    int transposedWidth = inputWidth + (inputWidth - 1) * (stride - 1);

    int virtualPad = filterSize - pad - 1;
    assert(virtualPad >= 0); // TODO: correctly compute it in cases where padding is more than full padding

    // Additional padding from left and up
    int padLeft = (outputWidth + 2 * pad - filterSize) % stride;
    int padUp = (outputHeight + 2 * pad - filterSize) % stride;

    int shiftLeft = -(virtualPad + padLeft);
    int shiftUp = -(virtualPad + padUp);

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
                            //std::cerr << "out(" << d << "," << ay / stride - shiftUp << "," << ax / stride - shiftLeft <<") "
                            //          << out(d, ay / stride - shiftUp, ax / stride - shiftLeft) << "+=" << input(fd, oy / stride, ox / stride) << "*" << filter(fd, d, fy, fx) << std::endl;
                            out(d, ay / stride - shiftUp, ax / stride - shiftLeft) += input(fd, oy / stride, ox / stride) * filter(fd, d, fy, fx);
                        }
                    }
                }
                //std::cerr << std::endl;
            }
        }
    }
}


}

#include "ConvolutionalLayer.hpp"
#include <cmath>

namespace MaskedCNN {

ConvolutionalLayer::ConvolutionalLayer(std::vector<int> dims, std::unique_ptr<Activation> activation, int stride,
                                       int filterSize, int pad, int featureMaps)
    :Layer(), activation(std::move(activation)), pad(pad),
      stride(stride), filterSize(filterSize), outputChannels(featureMaps)
{
    assert(dims.size() == 3);
    filterDepth = dims[0];
    inputHeight = dims[1];
    inputWidth = dims[2];

    outputWidth = std::floor((inputWidth + pad * 2 - filterSize) / (double)stride + 1);
    outputHeight = std::floor((inputHeight + pad * 2  - filterSize) / (double)stride + 1);

    weights.resize({outputChannels, filterSize, filterSize, filterDepth});
    weight_delta.resize({outputChannels, filterSize, filterSize, filterDepth});

    biases.resize({outputChannels});
    bias_delta.resize({outputChannels});

    z.resize({outputChannels, outputHeight, outputWidth});
    dy_dz.resize({outputChannels, outputHeight, outputWidth});
    delta.resize({outputChannels, outputHeight, outputWidth});
    output.resize({outputChannels, outputHeight, outputWidth});
}


void ConvolutionalLayer::forwardPropagate(const Tensor<float> &input)
{
    z.zero();

    for (int d = 0; d < outputChannels; d++)
    {
        for (int ay = 0, y = -pad; ay < outputHeight; ay++, y += stride)
        {
            for (int ax = 0, x = -pad; ax < outputWidth; ax++, x += stride)
            {
                float sum = 0;

                for (int fy = 0; fy < filterSize; fy++)
                {
                    for (int fx = 0; fx < filterSize; fx++)
                    {
                        int oy = y + (filterSize - fy - 1);
                        int ox = x + (filterSize - fx - 1);

                        if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                        {
                            for (int fd = 0; fd < input.dimensionCount(); fd++)
                            {
                                sum += weights(d,fy,fx,fd) * input(fd, oy, ox);
                            }
                        }
                    }
                }

                z(d, ay, ax) += sum + biases[d];
            }
        }
    }

    activation->activate(&z[0], &output[0], &dy_dz[0], output.elementCount());
}


void ConvolutionalLayer::backwardPropagate(const Tensor<float>& input, Tensor<float>& prevDelta)
{
    prevDelta.zero();
    weight_delta.zero();
    bias_delta.zero();

    elementwiseMultiplication(delta.dataAddress(), dy_dz.dataAddress(),
                              delta.dataAddress(), delta.elementCount());

    for (int d = 0; d < outputChannels; d++)
    {
        for (int fy = 0; fy < filterSize; fy++)
        {
            for (int ay = 0; ay < outputHeight; ay++)
            {
                int y = stride * ay + fy;
                if (y < 0 || y >= inputHeight) continue;

                for (int fx = 0; fx < filterSize; fx++)
                {
                    for (int ax = 0; ax < outputWidth; ax++)
                    {
                        int x = stride * ax + fx;
                        if (x < 0 || x >= inputWidth) continue;

                        float del = delta(d, ay, ax);

                        for (int fd = 0; fd < filterDepth; fd++)
                        {
                            weight_delta(d, fy, fx, fd) += del * input(fd, y, x);
                        }

                        bias_delta[d] += del;
                    }
                }
            }
        }
    }

    // Doing transposed convolution here
    // https://arxiv.org/pdf/1603.07285.pdf

    // "inserting" (stride-1) virtual zeroes in the transposed input (input to transposed convolution,
    // in this case, is our output) between each real element
    int transposedHeight = outputHeight + (outputHeight - 1) * (stride - 1);
    int transposedWidth = outputWidth + (outputWidth - 1) * (stride - 1);

    int virtualPad = filterSize - pad - 1;
    assert(virtualPad >= 0); // TODO: correctly compute it in cases where padding is more than full padding

    // Additional padding from left and up
    int padLeft = (inputWidth + 2 * pad - filterSize) % stride;
    int padUp = (inputHeight + 2 * pad - filterSize) % stride;

    int shiftLeft = -(virtualPad + padLeft);
    int shiftUp = -(virtualPad + padUp);

    for (int d = 0; d < outputChannels; d++)
    {
        for (int ay = shiftUp; ay < shiftUp + transposedHeight + padUp - filterSize - 1; ay++)
        {
            for (int fy = 0; fy < filterSize; fy++)
            {
                int oy = ay + fy;
                if ((oy % stride != 0) || oy < 0 || oy >= outputHeight) continue;
                for (int ax = shiftLeft; ax < shiftLeft + transposedWidth + padLeft - filterSize - 1; ax++)
                {
                    for (int fx = 0; fx < filterSize; fx++)
                    {
                        int ox = ax + fx;
                        if ((ox % stride != 0) || ox < 0 || ox >= outputWidth) continue;

                        for (int fd = 0; fd < input.dimensionCount(); fd++)
                        {
                            prevDelta(fd, oy / stride, ox / stride) += delta(d, oy, ox) * weights(d, fy, fx, fd);
                        }
                    }
                }
            }
        }
    }
}

std::vector<int> ConvolutionalLayer::getOutputDimensions()
{
    return {outputChannels, outputHeight, outputWidth};
}

int ConvolutionalLayer::getNeuronInputNumber() const
{
    return filterSize * filterSize * filterDepth;
}



}

#include "ConvolutionalLayer.hpp"
#include <cmath>

namespace MaskedCNN {

ConvolutionalLayer::ConvolutionalLayer(std::vector<int> dims, std::unique_ptr<Activation> activation, int stride,
                                       int filterSize, int featureMaps)
    :Layer(), activation(std::move(activation)),
      stride(stride), filterSize(filterSize), outputChannels(featureMaps)
{
    assert(dims.size() == 3);
    filterDepth = dims[0];
    inputHeight = dims[1];
    inputWidth = dims[2];
    outputWidth = std::floor((inputWidth /* + pad * 2 */ - filterSize) / (double)stride + 1);
    outputHeight = std::floor((inputHeight /* + pad * 2 */ - filterSize) / (double)stride + 1);

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
        for (int ay = 0, y = 0; ay < outputHeight; ay++, y += stride)
        {
            for (int ax = 0, x = 0; ax < outputWidth; ax++, x += stride)
            {
                float sum = 0;

                for (int fy = 0; fy < filterSize; fy++)
                {
                    for (int fx = 0; fx < filterSize; fx++)
                    {
                        int oy = y + (filterSize - fy - 1);
                        int ox = x + (filterSize - fx - 1);

                        for (int fd = 0; fd < input.dimensionCount(); fd++)
                        {
                            sum += weights(d,fy,fx,fd) * input(fd, oy, ox);
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

    // TODO: strides

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

    for (int d = 0; d < outputChannels; d++)
    {
        for (int ay = -filterSize + 1, y = 0; ay < outputHeight + filterSize - 1; ay++, y += 1)
        {
            for (int ax = -filterSize + 1, x = 0; ax < outputWidth + filterSize - 1; ax++, x += 1)
            {
                for (int fy = 0; fy < filterSize; fy++)
                {
                    for (int fx = 0; fx < filterSize; fx++)
                    {
                        int oy = ay + fy;
                        int ox = ax + fx;

                        if (oy >= 0 && oy < outputHeight && ox >= 0 && ox < outputWidth)
                        {
                            for (int fd = 0; fd < input.dimensionCount(); fd++)
                            {
                                prevDelta(fd, y, x) += delta(d, ay + fy, ax + fx) * weights(d, fy, fx, fd);
                            }
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

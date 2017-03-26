#include "ConvolutionalLayer.hpp"
#include <cmath>

namespace MaskedCNN {

ConvolutionalLayer::ConvolutionalLayer(int width, int height, int inputChannels, std::unique_ptr<Activation> activation, int stride,
                                       int filterSize, int featureMaps)
    :Layer(), activation(std::move(activation)),
      stride(stride), filterSize(filterSize), outputChannels(featureMaps)
{
    outputSizeX = std::floor((width /* + pad * 2 */ - filterSize) / (double)stride + 1);
    outputSizeY = std::floor((height /* + pad * 2 */ - filterSize) / (double)stride + 1);

    weights.resize({outputChannels, filterSize, filterSize, inputChannels});
    weight_delta.resize({outputChannels, filterSize, filterSize, inputChannels});

    biases.resize({outputChannels});
    bias_delta.resize({outputChannels});

    z.resize({outputChannels, outputSizeY, outputSizeX});
    dy_dz.resize({outputChannels, outputSizeY, outputSizeX});
    delta.resize({outputChannels, outputSizeY, outputSizeX});
    output.resize({outputChannels, outputSizeY, outputSizeX});
}


void ConvolutionalLayer::forwardPropagate(const Tensor<float> &input)
{
    for (int d = 0; d < output.dimensionCount(); d++)
    {
        for (int ay = 0; ay < output.columnLength(); ay++)
        {
            for (int ax = 0; ax < output.rowLength(); ax++)
            {
                int y = ay * stride;
                int x = ax * stride;

                double sum = 0;

                for (int fy = 0; fy < filterSize; fy++)
                {
                    for (int fx = 0; fx < filterSize; fx++)
                    {
                        int oy = y + fy;
                        int ox = x + fx;

                        for (int fd = 0; fd < input.dimensionCount(); fd++)
                        {
                            sum += weights(d,fy,fx,fd) * input(fd, oy, ox);
                        }
                    }
                }

                output(d, ay, ax) = sum;
            }
        }
    }
}


void ConvolutionalLayer::backwardPropagate(const Tensor<float>& input, Tensor<float>& prevDelta)
{
    prevDelta.zero();
    weight_delta.zero();
    bias_delta.zero();

    activation->activate(&z[0], &output[0], &dy_dz[0], output.elementCount());

    for (int d = 0; d < output.dimensionCount(); d++)
    {
        for (int ay = 0; ay < output.columnLength(); ay++)
        {
            for (int ax = 0; ax < output.rowLength(); ax++)
            {
                int y = ay * stride;
                int x = ax * stride;

                float de_dy = delta(d,ay,ax);

                for (int fy = 0; fy < filterSize; fy++)
                {
                    for (int fx = 0; fx < filterSize; fx++)
                    {
                        int oy = y + fy;
                        int ox = x + fx;

                        int rfy = filterSize - fy - 1;
                        int rfx = filterSize - fx - 1;

                        for (int fd = 0; fd < input.dimensionCount(); fd++)
                        {
                            weight_delta(d,rfy,rfx,fd) += de_dy * dy_dz(fd, oy, ox);
                            prevDelta(fd, oy, ox) += de_dy * weights(d,rfy,rfx,fd);
                        }
                    }
                }

                bias_delta[d] += de_dy;
            }
        }
    }
}

std::vector<int> ConvolutionalLayer::getOutputDimensions()
{
    return {outputChannels, outputSizeY, outputSizeX};
}



}

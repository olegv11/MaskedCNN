#include "ConvolutionalLayer.hpp"
#include "ConvOps.hpp"
#include <cmath>

namespace MaskedCNN {

BaseConvolutionalLayer::BaseConvolutionalLayer(std::unique_ptr<Activation> activation, int stride, int filterSize,
                                               int pad, int filterDepth, int featureMaps)
    :Layer(), activation(std::move(activation)), pad(pad),
      stride(stride), filterSize(filterSize), filterDepth(filterDepth), outputChannels(featureMaps)
{
    weights.resize({outputChannels, filterDepth, filterSize, filterSize});
    weight_delta.resize({outputChannels, filterDepth, filterSize, filterSize});

    biases.resize({outputChannels});
    bias_delta.resize({outputChannels});
}

std::vector<int> BaseConvolutionalLayer::getOutputDimensions()
{
    return {outputChannels, outputHeight, outputWidth};
}

int BaseConvolutionalLayer::getNeuronInputNumber() const
{
    return filterSize * filterSize * filterDepth;
}



ConvolutionalLayer::ConvolutionalLayer(std::unique_ptr<Activation> activation, int stride, int filterSize, int pad,
                                       int filterDepth, int featureMaps)
    : BaseConvolutionalLayer(std::move(activation), stride, filterSize, pad, filterDepth, featureMaps)
{

}

DeconvolutionalLayer::DeconvolutionalLayer(std::unique_ptr<Activation> activation, int stride, int filterSize, int pad,
                                       int filterDepth, int featureMaps)
    : BaseConvolutionalLayer(std::move(activation), stride, filterSize, pad, filterDepth, featureMaps)
{
    assert(pad == 0);
}

void DeconvolutionalLayer::forwardPropagate(const Tensor<float> &input)
{
    if (isTraining && !initDone)
    {
        initializeWeightsNormalDistrCorrectedVar();
        initDone = true;
    }

    auto dims = input.dimensions();
    assert(dims.size() == 3);
    assert(filterDepth == dims[0]);
    inputHeight = dims[1];
    inputWidth = dims[2];

    outputWidth = stride * (inputWidth - 1) + filterSize;
    outputHeight = stride * (inputHeight - 1) + filterSize;

    z.resize({outputChannels, outputHeight, outputWidth});
    dy_dz.resize({outputChannels, outputHeight, outputWidth});
    delta.resize({outputChannels, outputHeight, outputWidth});
    output.resize({outputChannels, outputHeight, outputWidth});

    transposedConvolution(input, weights, z, filterSize, stride, pad);
    for (int d = 0; d < outputChannels; d++)
    {
        for (int ay = 0; ay < outputHeight; ay++)
        {
            for (int ax = 0; ax < outputWidth; ax++)
            {
                z(d, ay, ax) += biases[d];
            }
        }
    }

    activation->activate(&z[0], &output[0], &dy_dz[0], output.elementCount());
}

void DeconvolutionalLayer::backwardPropagate(const Tensor<float> &input, Tensor<float> &prevDelta)
{
    (void)input;
    (void)prevDelta;
    assert(false);
}


void ConvolutionalLayer::forwardPropagate(const Tensor<float> &input)
{
    if (isTraining && !initDone)
    {
        initializeWeightsNormalDistrCorrectedVar();
        initDone = true;
    }

    auto dims = input.dimensions();
    assert(dims.size() == 3);
    assert(filterDepth == dims[0]);
    inputHeight = dims[1];
    inputWidth = dims[2];

    outputWidth = std::floor((inputWidth + pad * 2 - filterSize) / (double)stride + 1);
    outputHeight = std::floor((inputHeight + pad * 2  - filterSize) / (double)stride + 1);

    z.resize({outputChannels, outputHeight, outputWidth});
    dy_dz.resize({outputChannels, outputHeight, outputWidth});
    delta.resize({outputChannels, outputHeight, outputWidth});
    output.resize({outputChannels, outputHeight, outputWidth});

    convolution(input, weights, z, filterSize, stride, pad);
    for (int d = 0; d < outputChannels; d++)
    {
        for (int ay = 0; ay < outputHeight; ay++)
        {
            for (int ax = 0; ax < outputWidth; ax++)
            {
                z(d, ay, ax) += biases[d];
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

    for (int fy = 0; fy < filterSize; fy++)
    {
        for (int ay = 0; ay < outputHeight; ay++)
        {
            int y = stride * ay + rot180(fy, filterSize);
            if (y < 0 || y >= inputHeight) continue;

            for (int fx = 0; fx < filterSize; fx++)
            {
                for (int ax = 0; ax < outputWidth; ax++)
                {
                    int x = stride * ax + rot180(fx, filterSize);
                    if (x < 0 || x >= inputWidth) continue;


                    for (int d = 0; d < outputChannels; d++)
                    {
                        float del = delta(d, ay, ax);

                        for (int fd = 0; fd < filterDepth; fd++)
                        {
                            weight_delta(d, fd, fy, fx) += del * input(fd, y, x);
                        }

                        bias_delta[d] += del;
                    }
                }
            }
        }
    }

    transposedConvolution(delta, weights, prevDelta, filterSize, stride, pad);
}






}

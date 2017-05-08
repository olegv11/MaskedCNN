#include "ConvolutionalLayer.hpp"
#include "ConvOps.hpp"
#include <cmath>

namespace MaskedCNN {

BaseConvolutionalLayer::BaseConvolutionalLayer(std::unique_ptr<Activation> activation, int stride, int filterSize,
                                               int pad, int filterDepth, int featureMaps, std::string name)
    :Layer(), activation(std::move(activation)), pad(pad),
      stride(stride), filterSize(filterSize), filterDepth(filterDepth), outputChannels(featureMaps)
{
    this->name = name;
    weights.resize({outputChannels, filterDepth, filterSize, filterSize});
    weight_delta.resize({outputChannels, filterDepth, filterSize, filterSize});

    biases.resize({outputChannels});
    bias_delta.resize({outputChannels});
}

BaseConvolutionalLayer::BaseConvolutionalLayer(std::unique_ptr<Activation> activation, Tensor<float>&& weights, Tensor<float>&& biases,
                                               int stride, int pad, std::string name)
    :Layer(std::move(weights), std::move(biases), name), activation(std::move(activation)), pad(pad), stride(stride)
{
    auto dims = this->weights.dimensions();

    assert(dims.size() == 4);
    assert(dims[2] == dims[3]);

    outputChannels = dims[0];
    filterDepth = dims[1];
    filterSize = dims[2];
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
                                       int filterDepth, int featureMaps, std::string name)
    : BaseConvolutionalLayer(std::move(activation), stride, filterSize, pad, filterDepth, featureMaps, name)
{

}

ConvolutionalLayer::ConvolutionalLayer(std::unique_ptr<Activation> activation, Tensor<float>&& weights, Tensor<float>&& biases,
                                       int stride, int pad, std::string name)
    :BaseConvolutionalLayer(std::move(activation), std::move(weights), std::move(biases), stride, pad, name)
{

}

DeconvolutionalLayer::DeconvolutionalLayer(std::unique_ptr<Activation> activation, int stride, int filterSize, int pad,
                                       int filterDepth, int featureMaps, std::string name)
    : BaseConvolutionalLayer(std::move(activation), stride, filterSize, pad, filterDepth, featureMaps)
{
    assert(pad == 0);
    this->name = name;
}

DeconvolutionalLayer::DeconvolutionalLayer(std::unique_ptr<Activation> activation, Tensor<float>&& weights, Tensor<float>&& biases, int stride, int pad, std::string name)
    :BaseConvolutionalLayer(std::move(activation), std::move(weights), std::move(biases), stride, pad)
{
    assert(pad == 0);
    this->name = name;    
}

void ConvolutionalLayer::forwardPropagate()
{
    const Tensor<float> &input = *bottoms[0]->getOutput();

    if (!initDone)
    {
        if (isTraining)
        {
            initializeWeightsNormalDistrCorrectedVar();
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
        mask.resize({outputHeight, outputWidth});

        initDone = true;
    }


    if (maskEnabled)
    {
        const Tensor<float> &prevMask = *bottoms[0]->getMask();
        convolveMaskIm2Col(prevMask, mask, maskColBuffer, filterSize, stride, pad);
        convolutionIm2ColMasked(input, mask, weights, colBuffer, outBuffer, z, filterSize, stride, pad);

        activateOutBuffer();

        activation->activate(&z[0], &output[0], &dy_dz[0], output.elementCount());

    }
    else
    {
        convolutionIm2Col(input, weights, colBuffer, z, filterSize, stride, pad);

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

}

void ConvolutionalLayer::activateOutBuffer()
{
    auto outBufferData = outBuffer.dataAddress();
    auto outputData = z.dataAddress();
    const auto& maskData = mask.dataAddress();

    for (int d = 0; d < outputChannels; d++)
    {
        float bias = biases[d];
        for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < outputWidth; x++)
            {
                if (maskData[y * outputWidth + x] > 0)
                {
                    int index = x + outputWidth * (y + outputHeight * d);
                    outputData[index] = *outBufferData + bias;
                    outBufferData++;
                }
            }
        }
    }

    activation->activate(&z[0], &output[0], &dy_dz[0], output.elementCount());
}

void DeconvolutionalLayer::forwardPropagate()
{
    const Tensor<float> &input = *bottoms[0]->getOutput();

    if (!initDone)
    {
        if (isTraining)
        {
            initializeWeightsNormalDistrCorrectedVar();
        }

        auto dims = input.dimensions();
        assert(dims.size() == 3);
        assert(filterDepth == dims[0]);
        inputHeight = dims[1];
        inputWidth = dims[2];

        outputWidth = stride * (inputWidth - 1) + filterSize - 2 * pad;
        outputHeight = stride * (inputHeight - 1) + filterSize - 2 * pad;

        z.resize({outputChannels, outputHeight, outputWidth});
        dy_dz.resize({outputChannels, outputHeight, outputWidth});
        delta.resize({outputChannels, outputHeight, outputWidth});
        output.resize({outputChannels, outputHeight, outputWidth});

        initDone = true;
    }

    transposedConvolutionIm2Col(input, weights, colBuffer, z, filterSize, stride, pad);
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

void DeconvolutionalLayer::backwardPropagate()
{
    assert(false);
}

void ConvolutionalLayer::backwardPropagate()
{
    const Tensor<float> &input = *bottoms[0]->getOutput();
    Tensor<float> &prevDelta = *bottoms[0]->getDelta();

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

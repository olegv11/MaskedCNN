#include "SoftmaxLayer.hpp"

#include<cmath>

namespace MaskedCNN
{

void softmax(const float *__restrict__ x, float *__restrict__ y, int num);

SoftmaxLayer::SoftmaxLayer(int numClasses)
    :numClasses(numClasses)
{
    output.resize({numClasses});
}

void SoftmaxLayer::forwardPropagate(const Tensor<float>& input)
{
    assert(input.dimensions() == std::vector<int>{numClasses});
    softmax(&input[0], &output[0], numClasses);
}

void SoftmaxLayer::backwardPropagate(const Tensor<float>&input, Tensor<float>& prevDelta)
{
    (void)input;
    assert(prevDelta.dimensions() == std::vector<int>{numClasses});
    prevDelta.zero();

    for (int i = 0; i < numClasses; i++)
    {
        double indicator = (i == groundTruth) ? 1.0 : 0.0;
        prevDelta[i] = -(indicator - output[i]);
    }

    loss = -std::log(output[groundTruth]);
}

std::vector<int> SoftmaxLayer::getOutputDimensions()
{
    return {1};
}

double SoftmaxLayer::getLoss() const
{
    return loss;
}

void SoftmaxLayer::setGroundTruth(int i)
{
    groundTruth = i;
}


void softmax(const float *__restrict__ x, float *__restrict__ y, int num)
{
    float maxValue = x[0];
    for (int i = 1; i < num; i++)
    {
        if (x[i] > maxValue)
        {
            maxValue = x[i];
        }
    }

    float sum = 0;
    for (int i = 0; i < num; i++)
    {
        y[i] = std::exp(x[i] - maxValue);
        sum += y[i];
    }

    sum = 1.0 / sum;

    for (int i = 0; i < num; i++)
    {
        y[i] *= sum;
    }
}

}



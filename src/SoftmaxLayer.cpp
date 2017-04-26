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

void SoftmaxLayer::forwardPropagate()
{
    const Tensor<float> &input = *bottoms[0]->getOutput();
    std::cout << "Forward start" << name << std::endl;
    assert(input.dimensions() == std::vector<int>{numClasses});
    softmax(&input[0], &output[0], numClasses);
}

void SoftmaxLayer::backwardPropagate()
{
    Tensor<float> &prevDelta = *bottoms[0]->getDelta();
    assert(prevDelta.dimensions() == std::vector<int>{numClasses});
    prevDelta.zero();

    for (int i = 0; i < numClasses; i++)
    {
        double indicator = (i == groundTruth) ? 1.0 : 0.0;
        prevDelta[i] = -(indicator - output[i]);
    }

    loss = -std::log(output[groundTruth] + 1e-9);
}

std::vector<int> SoftmaxLayer::getOutputDimensions()
{
    return {1};
}

float SoftmaxLayer::getLoss() const
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

    float sum = 1e-9;
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

#include "Layer.hpp"
#include <random>
#include <cmath>

namespace MaskedCNN
{

static std::random_device rd;
static std::mt19937 gen(rd());

Layer::Layer()
{
}


Layer::Layer(Tensor<float> &&weights, Tensor<float> &&biases)
    :weights(std::move(weights)), biases(std::move(biases))
{
}


void Layer::setTrainer(std::unique_ptr<TrainingRegime> trainer)
{
    this->trainer.swap(trainer);
}

void Layer::setSGD(float learningRate, float l2Reg, int numBatch, int numData, float momentum)
{
    this->trainer = std::make_unique<StochasticGradientDescent>(
                learningRate, l2Reg, weights.elementCount(), biases.elementCount(),
                numBatch, numData, momentum);
}

void Layer::setAdaGrad(float learningRate, float l2Reg, int numBatch, int numData)
{
    this->trainer = std::make_unique<AdaGrad>(
                learningRate, l2Reg, weights.elementCount(), biases.elementCount(),
                numBatch, numData);

}

void Layer::setRMSProp(float learningRate, float l2Reg, int numBatch, int numData, float decay)
{
    this->trainer = std::make_unique<RmsProp>(
                learningRate, l2Reg, weights.elementCount(), biases.elementCount(),
                numBatch, numData, decay);

}

void Layer::updateParameters()
{
    if (trainer)
    {
        trainer->updateParameters(weights.dataAddress(), weight_delta.dataAddress(),
                                  biases.dataAddress(), bias_delta.dataAddress());
    }
}

const Tensor<float>* Layer::getOutput()
{
    return &output;
}

Tensor<float>* Layer::getDelta()
{
    return &delta;
}

void Layer::initializeWeightsStandardDistr()
{
    std::normal_distribution<double> d(0.0, 1.0);

    int weightCount = weights.elementCount();
    float *w = weights.dataAddress();
    for (int i = 0; i < weightCount; i++)
    {
        w[i] = d(gen);
    }
}

// See:
//Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
void Layer::initializeWeightsNormalDistrCorrectedVar()
{
    std::normal_distribution<double> d(0.0, std::sqrt(2.0 / weights.elementCount()));

    int weightCount = weights.elementCount();
    float *w = weights.dataAddress();
    for (int i = 0; i < weightCount; i++)
    {
        w[i] = d(gen);
    }
}

}

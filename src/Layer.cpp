#include "Layer.hpp"
#include <random>

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

void Layer::initializeWeightsNormalDistr()
{
    std::normal_distribution<double> d(0.0, 1.0);

    int weightCount = weights.elementCount();
    float *w = weights.dataAddress();
    for (int i = 0; i < weightCount; i++)
    {
        w[i] = d(gen);
        std::cout << "W " << w[i] << std::endl;
    }
}

}

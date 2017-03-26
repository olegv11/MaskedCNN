#include "Network.hpp"
#include <iostream>
#include <memory>

using namespace MaskedCNN;

std::vector<std::unique_ptr<Layer>> layers(4);
void trainOnExample(Tensor<float> *example, int groundTruth);

int main()
{
    layers[0].reset(new InputLayer({2}));
    layers[1].reset(new FullyConnectedLayer(2, 10, std::make_unique<Sigmoid>()));
    layers[1]->setSGD(0.001, 0, 4, 4);
    layers[1]->initializeWeightsNormalDistr();
    layers[2].reset(new FullyConnectedLayer(10, 2, std::make_unique<Sigmoid>()));
    layers[2]->setSGD(0.001, 0, 4, 4);
    layers[2]->initializeWeightsNormalDistr();
    layers[3].reset(new SoftmaxLayer(2));

    Tensor<float> example0({2});
    example0[0] = 0; example0[1] = 0;

    Tensor<float> example1({2});
    example1[0] = 0; example1[1] = 1;

    Tensor<float> example2({2});
    example2[0] = 1; example2[1] = 0;

    Tensor<float> example3({2});
    example3[0] = 1; example3[1] = 1;

    for (int epoch = 0; epoch < 100000; epoch++)
    {
        trainOnExample(&example0, 0);
        trainOnExample(&example1, 1);
        trainOnExample(&example2, 1);
        trainOnExample(&example3, 0);
    }

    return 0;
}

void trainOnExample(Tensor<float> *example, int groundTruth)
{
    SoftmaxLayer *softmax = dynamic_cast<SoftmaxLayer*>(layers[layers.size() - 1].get());
    softmax->setGroundTruth(groundTruth);
    const Tensor<float> *data = example;
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        layers[i]->forwardPropagate(*data);
        data = layers[i]->getOutput();
    }

    for (uint32_t i = layers.size() - 1; i > 0; i--)
    {
        layers[i]->backwardPropagate(*layers[i-1]->getOutput(), *layers[i-1]->getDelta());
    }

    for (uint32_t i = 0; i < layers.size(); i++)
    {
        layers[i]->updateParameters();
    }

    std::cout << softmax->getLoss() << std::endl;
}

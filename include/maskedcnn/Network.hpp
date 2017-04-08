#pragma once
#include "Layer.hpp"
#include "InputLayer.hpp"
#include "ConvolutionalLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "PoolLayer.hpp"
#include "SoftmaxLayer.hpp"
#include "Activation.hpp"
#include "TrainingRegime.hpp"
#include "DataLoader.hpp"


#include <vector>

namespace MaskedCNN
{

class Network
{
public:
    Network();

private:
    std::vector<Layer> layers;
};

}

void createNetwork(int miniBatchSize, int exampleCount);
void train(MaskedCNN::CIFARDataLoader &loader, int miniBatchSize);

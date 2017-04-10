#include "Network.hpp"
#include <iostream>
#include <memory>
#include <algorithm>
#include <random>
#include <iomanip>

using namespace MaskedCNN;

std::vector<std::unique_ptr<Layer>> layers(10);



int main()
{
    CIFARDataLoader loader("/home/oleg/Deep_learning/CIFAR-100/");
    loader.loadSmallData();

    createNetwork(20, loader.trainCount());
    train(loader, 20);

    return 0;
}

void createNetwork(int miniBatchSize, int exampleCount)
{
    float step = 0.00004;
    //float l2 = 0.0001;
    float l2 = 0;

    layers[0].reset(new InputLayer({3,32,32}));

    layers[1].reset(new ConvolutionalLayer(layers[0]->getOutputDimensions(), std::make_unique<ReLu>(), 1, 3, 2, 16));
    layers[1]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);
    layers[1]->initializeWeightsNormalDistrCorrectedVar();

    layers[2].reset(new PoolLayer(layers[1]->getOutputDimensions(), 2, 2));

    layers[3].reset(new ConvolutionalLayer(layers[2]->getOutputDimensions(), std::make_unique<ReLu>(), 1, 3, 2, 32));
    layers[3]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);
    layers[3]->initializeWeightsNormalDistrCorrectedVar();

    layers[4].reset(new ConvolutionalLayer(layers[3]->getOutputDimensions(), std::make_unique<ReLu>(), 1, 3, 2, 64));
    layers[4]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);
    layers[4]->initializeWeightsNormalDistrCorrectedVar();

    layers[5].reset(new ConvolutionalLayer(layers[4]->getOutputDimensions(), std::make_unique<ReLu>(), 1, 3, 2, 128));
    layers[5]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);
    layers[5]->initializeWeightsNormalDistrCorrectedVar();

    layers[6].reset(new PoolLayer(layers[5]->getOutputDimensions(), 2, 2));

    auto x = layers[6]->getOutputDimensions();
    layers[7].reset(new FullyConnectedLayer(x[0]*x[1]*x[2], 1024, std::make_unique<ReLu>()));
    layers[7]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);
    layers[7]->initializeWeightsNormalDistrCorrectedVar();


    auto y = layers[7]->getOutputDimensions();
    layers[8].reset(new FullyConnectedLayer(y[0]*y[1]*y[2], 2, std::make_unique<ReLu>()));
    layers[8]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);
    layers[8]->initializeWeightsNormalDistrCorrectedVar();

    layers[9].reset(new SoftmaxLayer(2));
}

void train(CIFARDataLoader& loader, int miniBatchSize)
{
    SoftmaxLayer *softmax = dynamic_cast<SoftmaxLayer*>(layers[layers.size() - 1].get());
    auto& trainData = loader.getTrainImages();
    auto& trainLabels = loader.getTrainLabels();

    normalize(trainData);

    std::vector<int> indices(trainData.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < 10000; epoch++)
    {
        for (unsigned int i = 0; i < trainData.size() / miniBatchSize; i++)
        {
            double sum = 0;
            std::random_shuffle(indices.begin(), indices.end());

            for (int j = 0; j < miniBatchSize; j++)
            {
                int index = indices[i * miniBatchSize + j];
                softmax->setGroundTruth(trainLabels[index]);
                const Tensor<float> *data = &trainData[index];

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
                sum += softmax->getLoss();
            }
            sum /= miniBatchSize;
            std::cout << std::setprecision(10) << sum << std::endl;
        }
        std::cout << "EPOCH " << epoch << " DONE" << std::endl;
    }

}

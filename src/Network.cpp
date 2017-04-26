#include "Network.hpp"
#include <string>
#include <iostream>
#include <memory>
#include <algorithm>
#include <random>
#include <iomanip>

#include <fstream>
#include "LayerLoader.hpp"

using namespace MaskedCNN;

std::vector<std::unique_ptr<Layer>> layers;



int main()
{
//    CIFARDataLoader loader("/home/oleg/Deep_learning/CIFAR-100/");
//    loader.loadSmallData();

//    createNetwork(25, loader.trainCount());
//    train(loader, 25);

    Tensor<float> image = loadImage("/home/oleg/image.jpeg");
    image.add(-104.00699, -116.66877, -122.67892);

    layers = loadCaffeNet("/home/oleg/Deep_learning/fcn/fcn.berkeleyvision.org/voc-fcn32s/fcn32s-heavy-pascal.caffemodel", 320, 240, 3);
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        layers[i]->setTrainingMode(false);
    }

    dynamic_cast<InputLayer*>(layers[0].get())->setInput(image);

    for (uint32_t i = 0; i < layers.size(); i++)
    {
        auto& l = *layers[i];
        l.forwardPropagate();
        auto x = l.getOutputDimensions();
        for (int j = 0; j < x.size(); j++)
        {
            std::cout << x[j] << " ";
        }

        if (l.getName() == "fc6" || l.getName() == "fc7")
        {
            std::cout << l.getOutput()->operator()(101,2,3) << std::endl;
        }
        else if (l.getName() == "score_fr")
        {
            std::cout << l.getOutput()->operator()(10,1,6) << " " << l.getOutput()->operator()(10,2,6) << std::endl;
        }
        else if (l.getName() == "upscore")
        {
            std::cout << l.getOutput()->operator()(5,20,20) << " " << l.getOutput()->operator()(15,50,190) << std::endl;
        }
        else
        {
            std::cout << l.getOutput()->operator()(2,22,23) << std::endl;
        }
    }

    const Tensor<float> *data = layers[layers.size() - 2]->getOutput();

    std::ofstream out("test");

    for (int y = 0; y < data->columnLength(); y++)
    {
        for (int x = 0; x < data->rowLength(); x++)
        {
            int max = 0;
            float maxel = data->operator()(0, y, x);
            for (int c = 0; c < data->channelLength(); c++)
            {
                float el = data->operator()(c, y, x);
                if (el > maxel)
                {
                    max = c;
                    maxel = data->operator()(c, y, x);
                    std::cout << "HOLY SHIT" << std::endl;
                }
            }
            out << max << " ";
        }
        out << std::endl;
    }

    return 0;
}

void setSGD(int miniBatchSize, int exampleCount, float step, float l2, float momentum = 0.9)
{
    for (size_t i = 1; i < layers.size() - 1; i++)
    {
        layers[i]->setSGD(step, l2, miniBatchSize, exampleCount, momentum);
    }
}

void setRMSProp(int miniBatchSize, int exampleCount, float step, float l2, float gamma = 0.9)
{
    for (size_t i = 1; i < layers.size() - 1; i++)
    {
        layers[i]->setRMSProp(step, l2, miniBatchSize, exampleCount, gamma);
    }
}

//void createNetwork(int miniBatchSize, int exampleCount)
//{
//    float step = 0.00002;
//    //float l2 = 0.0001;
//    float l2 = 0;

//    layers[0].reset(new InputLayer({3,32,32}));

//    layers[1].reset(new ConvolutionalLayer(std::make_unique<ReLu>(), 1, 3, 2, 3, 16));
//    layers[1]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);

//    layers[2].reset(new PoolLayer(2, 2));

//    layers[3].reset(new ConvolutionalLayer(std::make_unique<ReLu>(), 1, 3, 2, 16, 32));
//    layers[3]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);

//    layers[4].reset(new ConvolutionalLayer(std::make_unique<ReLu>(), 1, 3, 2, 32, 64));
//    layers[4]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);

//    layers[5].reset(new ConvolutionalLayer(std::make_unique<ReLu>(), 1, 3, 2, 64, 128));
//    layers[5]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);

//    layers[6].reset(new PoolLayer(2, 2));

//    layers[7].reset(new FullyConnectedLayer(std::make_unique<ReLu>(), 1024));
//    layers[7]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);


//    layers[8].reset(new DropoutLayer(0.5));

//    layers[9].reset(new FullyConnectedLayer(std::make_unique<ReLu>(), 2));
//    layers[9]->setRMSProp(step, l2, miniBatchSize, exampleCount, 0.9);

//    layers[10].reset(new SoftmaxLayer(2));
//}

//void train(CIFARDataLoader& loader, int miniBatchSize)
//{
//    SoftmaxLayer *softmax = dynamic_cast<SoftmaxLayer*>(layers[layers.size() - 1].get());
//    auto& trainData = loader.getTrainImages();
//    auto& trainLabels = loader.getTrainLabels();

//    normalize(trainData);

//    std::vector<int> indices(trainData.size());
//    std::iota(indices.begin(), indices.end(), 0);

//    for (unsigned int i = 0; i < layers.size(); i++)
//    {
//        layers[i]->setTrainingMode(true);
//    }

//    for (int epoch = 0; epoch < 10000; epoch++)
//    {
//        std::random_shuffle(indices.begin(), indices.end());
//        for (unsigned int i = 0; i < trainData.size() / miniBatchSize; i++)
//        {
//            double sum = 0;

//            for (int j = 0; j < miniBatchSize; j++)
//            {
//                int index = indices[i * miniBatchSize + j];
//                softmax->setGroundTruth(trainLabels[index]);
//                const Tensor<float> *data = &trainData[index];

//                for (uint32_t i = 0; i < layers.size(); i++)
//                {
//                    layers[i]->forwardPropagate(*data);
//                    data = layers[i]->getOutput();
//                }

//                for (uint32_t i = layers.size() - 1; i > 0; i--)
//                {
//                    layers[i]->backwardPropagate(*layers[i-1]->getOutput(), *layers[i-1]->getDelta());
//                }

//                for (uint32_t i = 0; i < layers.size(); i++)
//                {
//                    layers[i]->updateParameters();
//                }
//                sum += softmax->getLoss();
//            }
//            sum /= miniBatchSize;
//            std::cout << std::setprecision(10) << sum << std::endl;
//        }
//        std::cout << "EPOCH " << epoch << " DONE" << std::endl;
//    }

//}

#include "Network.hpp"
#include "Visuals.hpp"
#include "DataLoader.hpp"
#include "Statistics.hpp"
#include <iostream>
#include <fstream>
#include <ostream>

using namespace MaskedCNN;

void fullAccuracyTest(std::string filename);
double testAccuracy(Network& net, std::vector<YoutubeMasksDataLoader::Item>& items);

double speedTest(Network& net, int percent);
void fullSpeedTest(Network &net, std::string filename);
void layerSpeedTest();
double layerSpeedTest(Network& net, int percent, int layers);
void fullLayerSpeedTest(Network& net, std::string filename, int layers);

int main()
{
    std::string filename = "/home/oleg/cctv.avi";
    std::string noisy = filename + ".noisy.avi";
    DenoiseVideo(filename, 3);
    //layerSpeedTest();
    return 0;
}


void layerSpeedTest()
{
    constexpr int layerNumber = 256;
    std::vector<std::unique_ptr<Layer>> layers;
    layers.emplace_back(std::make_unique<InputLayer>("InputBig"));
    layers.emplace_back(std::make_unique<ConvolutionalLayer>(std::make_unique<Id>(), 1, 3, 0, layerNumber, 3, "Convolve"));
    layers[1]->addBottom(layers[0].get());

    std::cout << "ConvLayer" << std::endl;
    Network n1(std::move(layers), 0);
    fullLayerSpeedTest(n1, "ConvolutionalLayerSpeedTestBig", layerNumber);

    layers.clear();
    layers.emplace_back(std::make_unique<InputLayer>("Input"));
    layers.emplace_back(std::make_unique<PoolLayer>(3, "Pool"));
    layers[1]->addBottom(layers[0].get());

    std::cout << "PoolLayer" << std::endl;
    Network n2(std::move(layers), 0);
    fullLayerSpeedTest(n2, "PoolLayerSpeedTestBig", layerNumber);

    layers.clear();
    layers.emplace_back(std::make_unique<InputLayer>("Input"));
    layers.emplace_back(std::make_unique<DeconvolutionalLayer>(std::make_unique<Id>(), 1, 3, 0, layerNumber, layerNumber, "Convolve"));
    layers[1]->addBottom(layers[0].get());

    std::cout << "DeconvLayer" << std::endl;
    Network n3(std::move(layers), 0);
    fullLayerSpeedTest(n3, "DeconvLayerSpeedTestBig", layerNumber);
}



void fullSpeedTest(Network& net, std::string filename)
{
    std::ofstream resultFile(filename);

    net.setMaskEnabled(false);
    net.setThreshold(0);

    double time = speedTest(net, 0);
    resultFile << "baseline," << time << "\n";
    std::cout << "baseline," << time << "\n";

    for (int i = 0; i <= 100; i += 10)
    {
        net.setMaskEnabled(true);
        time = speedTest(net, i);
        resultFile << i << "," << time << "\n";
        std::cout << i << "," << time << "\n";
        std::cout << std::flush;
    }

}

void fullLayerSpeedTest(Network& net, std::string filename, int layers)
{
    std::ofstream resultFile(filename);

    net.setMaskEnabled(false);
    net.setThreshold(0);

    double time;
    time = layerSpeedTest(net, 0, layers);
    resultFile << "baseline," << time << "\n";
    std::cout << "baseline," << time << "\n";

    for (int i = 0; i <= 100; i += 10)
    {
        net.setMaskEnabled(true);
        time = layerSpeedTest(net, i, layers);
        resultFile << i << "," << time << "\n";
        std::cout << i << "," << time << "\n";
        std::cout << std::flush;
    }

}

double layerSpeedTest(Network& net, int percent, int layers)
{
    constexpr int numberOfTests = 500;

    Tensor<float> image(std::vector<int>{layers, 300, 300});
    Tensor<float> mask(std::vector<int>{300,300});
    mask.fillwith(0);

    double sumTime = 0;

    for (int y = 0; y < mask.columnLength() * std::sqrt(percent / 100); y++)
    {
        for (int x = 0; y < mask.rowLength() * std::sqrt(percent / 100); y++)
        {
            mask(y, x) = 1;
        }
    }

    for (int i = 0; i < numberOfTests; i++)
    {
        net.dummyForward(image, mask);
        sumTime += net.forwardTime();
        changePercentOfFrame(mask, percent);
    }

    return sumTime / numberOfTests;
}

double speedTest(Network& net, int percent)
{
    constexpr int numberOfTests = 1000;

    cv::Mat image = cv::Mat::zeros(1000, 1000, CV_8UC3);

    double sumTime = 0;

    for (int i = 0; i < numberOfTests; i++)
    {
        net.forward(image);
        sumTime += net.forwardTime();
        changePercentOfFrame(image, percent);
    }

    return sumTime / numberOfTests;
}

void fullAccuracyTest(std::string filename)
{
    Network net("/home/oleg/Deep_learning/fcn/fcn.berkeleyvision.org/voc-fcn32s/fcn32s-heavy-pascal.caffemodel", 0);
    YoutubeMasksDataLoader loader("/home/oleg/Загрузки/youtube_masks/");
    auto items = loader.loadAllItems();
    std::ofstream resultFile(filename);

    net.setMaskEnabled(false);
    auto iou = testAccuracy(net, items);

    resultFile << "baseline," << iou << std::endl;
    std::cout << "baseline," << iou << std::endl;

    for (int threshold = 0; threshold <= 100; threshold += 10)
    {
        net.setMaskEnabled(true);
        net.setThreshold(threshold);

        iou = testAccuracy(net, items);
        resultFile << threshold << "," << iou << std::endl;
        std::cout << threshold << "," << iou << std::endl;
    }
}

double testAccuracy(Network& net, std::vector<YoutubeMasksDataLoader::Item>& items)
{
    std::vector<std::tuple<int, Tensor<float>, Tensor<float>>> results;
    int i = 0;
    for (auto& item: items)
    {
        Tensor<float> label = labelToTensor(item.mask, item.label);
        if ((label.rowLength() != item.image.cols) || (label.columnLength() != item.image.rows))
        {
            std::cerr << "Dimension mismatch!" << label.rowLength() << "x" <<label.columnLength()
                      << "vs " << item.image.cols <<"x"<<item.image.rows<< std::endl;
            continue;
        }
        net.forward(item.image);
        Tensor<float> result = net.getOutput();

        results.emplace_back(item.label, result, label);
        if (i % 100 == 0)
        {
            std::cout << "Image " << i << " done" << std::endl;
        }
        i++;
    }

    return IoU(results);
}

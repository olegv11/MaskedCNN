#pragma once
#include <string>
#include <vector>
#include "Tensor.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace MaskedCNN
{

class CIFARDataLoader
{
public:
    CIFARDataLoader(const std::string& path);
    void loadData();
    void loadSmallData();

    std::vector<Tensor<float>>& getTrainImages() { return trainImages; }
    std::vector<Tensor<float>>& getTestImages() { return testImages; }

    std::vector<int>& getTrainLabels() { return trainLabels; }
    std::vector<int>& getTestLabels() { return testLabels; }

    int trainCount() const { return trainImages.size(); }

private:
    void loadLabels();
    void loadTestData();
    void loadTrainData();
    void loadTrainDataSmall();

    void normalizeData();

    std::string basePath;
    std::vector<std::string> labels;

    std::vector<Tensor<float>> trainImages;
    std::vector<int> trainLabels;
    std::vector<Tensor<float>> testImages;
    std::vector<int> testLabels;
};


class YoutubeMasksDataLoader
{
public:
    YoutubeMasksDataLoader(std::string path);
    struct Item
    {
        int label;
        cv::Mat image;
        cv::Mat mask;
    };

    std::vector<Item> loadAllItems();


private:
    void loadLabels();
    std::string path;
    std::vector<std::string> labels;
};

}

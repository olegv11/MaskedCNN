#include "DataLoader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace MaskedCNN
{

CIFARDataLoader::CIFARDataLoader(const std::string& path)
    :basePath(path)
{

}

void CIFARDataLoader::loadLabels()
{
    std::ifstream labelFile(basePath + "labels.txt");
    std::string line;

    while (labelFile)
    {
        std::getline(labelFile, line);
        if (!line.empty())
        {
            labels.push_back(line);
        }
    }
}

void CIFARDataLoader::loadTestData()
{
    std::ifstream testFile(basePath + "test.txt");
    std::string line;

    while (testFile)
    {
        std::getline(testFile, line);
        if (!line.empty())
        {
            std::istringstream sStream(line);
            std::string filename;
            int label;
            sStream >> filename >> label;

            testImages.push_back(loadImage(basePath + filename));
            testLabels.push_back(label);
        }
    }
}

void CIFARDataLoader::loadTrainData()
{
    std::ifstream trainFile(basePath + "train.txt");
    std::string line;

    while (trainFile)
    {
        std::getline(trainFile, line);
        if (!line.empty())
        {
            std::istringstream sStream(line);
            std::string filename;
            int label;
            sStream >> filename >> label;

            trainImages.push_back(loadImage(basePath + filename));
            trainLabels.push_back(label);
        }
    }
}

void CIFARDataLoader::loadTrainDataSmall()
{
    std::ifstream trainFile(basePath + "train.txt");
    std::string line;

    int categoriesLoaded[2] = {0,0};

    while (trainFile)
    {
        std::getline(trainFile, line);
        if (!line.empty())
        {
            std::istringstream sStream(line);
            std::string filename;
            int label;
            sStream >> filename >> label;
            if (label > 1) continue;
            if (categoriesLoaded[label] >= 25) continue;
            categoriesLoaded[label]++;
            trainImages.emplace_back(loadImage(basePath + filename));
            trainLabels.push_back(label);
        }
    }
}

void CIFARDataLoader::normalizeData()
{

}

void CIFARDataLoader::loadData()
{
    loadLabels();
    loadTrainData();
    loadTestData();
}

void CIFARDataLoader::loadSmallData()
{
    loadLabels();
    loadTrainDataSmall();
}

Tensor<float> loadImage(const std::string& path)
{
    cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_COLOR);

    //cv::Rect rect(110, 70, 150, 150);
    //cv::Mat cropped = image(rect);

    cv::Mat BGR[3];
    cv::split(image, BGR);

    Tensor<float> result(std::vector<int>({3, image.rows, image.cols}));

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            result(0, y, x) = BGR[0].at<uchar>(y,x);
            result(1, y, x) = BGR[1].at<uchar>(y,x);
            result(2, y, x) = BGR[2].at<uchar>(y,x);
        }
    }

    return result;
}




}

#include "DataLoader.hpp"
#include "Visuals.hpp"
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



}

#include "Statistics.hpp"

namespace MaskedCNN {


double IoU(std::vector<std::tuple<int, Tensor<float>, Tensor<float>>> results)
{
    std::vector<int> correctlyClassified(20);
    std::vector<int> incorrectlyClassified(20);
    std::vector<int> overallPixels(20);

    for (const auto& resultTuple: results)
    {
        int label = std::get<0>(resultTuple);
        const auto& resultImage = std::get<1>(resultTuple);
        const auto& resultMask = std::get<2>(resultTuple);

        int rows = resultImage.columnLength();
        int cols = resultImage.rowLength();

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                if (resultMask(y,x) == label && resultImage(y,x) == label)
                {
                    correctlyClassified[resultMask(y,x)]++;
                }
                else if (resultMask(y,x) != resultImage(y,x))
                {
                    incorrectlyClassified[resultMask(y,x)]++;
                    if (resultMask(y,x) != label && resultMask(y,x) != 0)
                    {
                        std::cerr << "WHAA" << resultMask(y,x);
                    }
                }
                else
                {
                    correctlyClassified[0]++;
                }

                overallPixels[resultMask(y,x)]++;
            }
        }
    }

    double sum = 0;
    int numClasses = 0;
    for (int i = 0; i < overallPixels.size(); i++)
    {
        if (overallPixels[i] > 0)
        {
            sum += (double) correctlyClassified[i] / (double)(overallPixels[i] + incorrectlyClassified[i]);
            numClasses++;
        }
    }
    return sum / numClasses;
}

void changePercentOfFrame(cv::Mat currentFrame, int percent)
{
    assert(currentFrame.type() == CV_8UC3);
    double part = percent / 100.0;
    int widthChange = currentFrame.cols * std::sqrt(part);
    int heightChange = currentFrame.rows * std::sqrt(part);

    for (int y = 0; y < heightChange; y++)
    {
        for (int x = 0; x < widthChange; x++)
        {
            cv::Vec3b& v = currentFrame.at<cv::Vec3b>(y,x);
            if (v[0] == 0)
            {
                v[0] = 255;
                v[1] = 255;
                v[2] = 255;
            }
            else
            {
                v[0] = 0;
                v[1] = 0;
                v[2] = 0;
            }
        }
    }
}

void changePercentOfFrame(Tensor<float> currentFrame, int percent)
{
    double part = percent / 100.0;
    int widthChange = currentFrame.rowLength() * std::sqrt(part);
    int heightChange = currentFrame.columnLength() * std::sqrt(part);

    for (int z = 0; z < currentFrame.channelLength(); z++)
    {
        for (int y = 0; y < heightChange; y++)
        {
            for (int x = 0; x < widthChange; x++)
            {
                float& el = currentFrame(z,y,x);
                if (el == 0)
                {
                    el = 255;
                }
                else
                {
                    el = 0;
                }
            }
        }
    }
}

}

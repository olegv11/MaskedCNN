#include "Visuals.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace MaskedCNN {

Tensor<float> matToTensor(const cv::Mat& image)
{
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

cv::Mat maskToMat(const Tensor<float>& tensor)
{
    cv::Mat image(tensor.columnLength(), tensor.rowLength(), CV_8UC1);

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            if (tensor(y,x) > 0)
            {
                image.at<unsigned char>(y,x) = 255;
            }
            else
            {
                image.at<unsigned char>(y,x) = 0;
            }

        }
    }

    return image;
}

cv::Mat cropLike(const cv::Mat data, const cv::Mat templateImage, int offset)
{
    cv::Rect rect(offset, offset, templateImage.cols, templateImage.rows);
    cv::Mat cropped(data(rect));

    cv::Mat result;
    cropped.copyTo(result);
    return cropped;
}

cv::Mat visualizeOutput(const Tensor<float>& tensor)
{
    cv::Mat image(tensor.columnLength(), tensor.rowLength(), CV_8UC3);

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            if (tensor(y,x) == 15) // person
            {
                image.at<cv::Vec3b>(y,x)[2] = 255;
                image.at<cv::Vec3b>(y,x)[1] = 0;
                image.at<cv::Vec3b>(y,x)[0] = 0;
            }
            else
            {
                image.at<cv::Vec3b>(y,x)[2] = 0;
                image.at<cv::Vec3b>(y,x)[1] = 0;
                image.at<cv::Vec3b>(y,x)[0] = 0;
            }

        }
    }

    return image;
}

Tensor<float> loadImage(const std::string& path)
{
    cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_COLOR);

    return matToTensor(image);
}

Tensor<float> maxarg(const Tensor<float>& data)
{
    Tensor<float> result({data.columnLength(), data.rowLength()});

    for (int y = 0; y < data.columnLength(); y++)
    {
        for (int x = 0; x < data.rowLength(); x++)
        {
            int max = 0;
            float maxel = data(0, y, x);
            for (int c = 0; c < data.channelLength(); c++)
            {
                float el = data(c, y, x);
                if (el > maxel)
                {
                    max = c;
                    maxel = data(c, y, x);
                }
            }

            result(y, x) = max;
        }
    }

    return result;
}


Tensor<float> diffFrames(const cv::Mat frame, const cv::Mat prevFrame, int threshold)
{
    Tensor<float> mask(std::vector<int>{prevFrame.rows, prevFrame.cols});
    cv::Mat diff = cv::Mat::zeros(frame.rows, frame.cols, CV_32SC3);
    cv::absdiff(frame, prevFrame, diff);

    for (int j = 0; j < frame.rows; j++)
    {
        for (int i = 0; i < frame.cols; i++)
        {
            cv::Vec3b x = diff.at<cv::Vec3b>(j,i);
            if (x[0] + x[1] + x[2] > threshold)
            {
                mask(j,i) = 255;
            }
            else
            {
                mask(j,i) = 0;
            }
        }
    }

    return mask;
}

}

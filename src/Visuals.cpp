#include "Visuals.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>


static std::random_device rd;
static std::mt19937 gen(rd());
std::uniform_real_distribution<double> distr(0.0, 1.0);

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

Tensor<float> labelToTensor(const cv::Mat& mask, int label)
{
    Tensor<float> result({mask.rows, mask.cols});
    if (mask.type() != CV_8UC3)
    {
        throw std::exception();
    }

    for (int y = 0; y < mask.rows; y++)
    {
        for (int x = 0; x < mask.cols; x++)
        {
            if (mask.at<cv::Vec3b>(y,x)[0] > 100)
            {
                result(y,x) = label;
            }
            else
            {
                result(y,x) = 0;
            }
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

Tensor<float> cropLike(const Tensor<float> data, const cv::Mat templateImage, int offset)
{
    int rows = templateImage.rows;
    int cols = templateImage.cols;

    Tensor<float> result({rows, cols});

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            result(y, x) = data(y + offset, x + offset);
        }
    }

    return result;
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


Tensor<float> diffFrames(const cv::Mat frame, const cv::Mat prevFrame, Tensor<float> accumMatrix, int threshold)
{
    Tensor<float> mask(std::vector<int>{prevFrame.rows, prevFrame.cols});
    cv::Mat diff = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
    cv::absdiff(frame, prevFrame, diff);

    for (int j = 0; j < frame.rows; j++)
    {
        for (int i = 0; i < frame.cols; i++)
        {
            cv::Vec3b x = diff.at<cv::Vec3b>(j,i);
            accumMatrix(j,i) += x[0] + x[1] + x[2];
            if (accumMatrix(j,i) > threshold)
            {
                mask(j,i) = 255;
                accumMatrix(j,i) = 0;
            }
            else
            {
                mask(j,i) = 0;
            }
        }
    }

    return mask;
}

void addNoiseToVideo(std::string filename, double prob)
{
    cv::VideoCapture cap(filename);
    cv::VideoWriter writer(filename + ".noisy.avi", CV_FOURCC('D','I','V','X'), 30, cv::Size(640, 360));
    if (!writer.isOpened())
    {
        throw std::exception();
    }
    cv::Mat frame;
    cv::Mat noise;
    while (cap.read(frame))
    {
        noise = saltAndPepper(frame, prob);
        writer << noise;
    }
}

cv::Mat saltAndPepper(const cv::Mat frame, double prob)
{
    cv::Mat noisy(frame.rows, frame.cols, CV_8UC3);
    frame.copyTo(noisy);

    for (int j = 0; j < frame.rows; j++)
    {
        for (int i = 0; i < frame.cols; i++)
        {
            cv::Vec3b &x = noisy.at<cv::Vec3b>(j,i);
            double var = distr(gen);
            if (var < prob)
            {
                x[0] = 0;
                x[1] = 0;
                x[2] = 0;
            }
            else if (var > 1 - prob)
            {
                x[0] = 255;
                x[1] = 255;
                x[2] = 255;
            }
        }
    }

    return noisy;
}

void DenoiseVideo(std::string filename, int window)
{
    cv::VideoCapture cap(filename);
    cv::VideoWriter writer(filename + ".denoised.avi", CV_FOURCC('D','I','V','X'), 30, cv::Size(640, 360));
    if (!writer.isOpened())
    {
        throw std::exception();
    }
    cv::Mat frame;
    cv::Mat denoised;
    while (cap.read(frame))
    {
        denoised = median(frame, window);
        writer << frame;
    }
}


cv::Mat median(const cv::Mat frame, int window)
{
    cv::Mat denoised(frame.rows, frame.cols, CV_8UC3);
    cv::medianBlur(frame, denoised, window);
    return denoised;
}

}

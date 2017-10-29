#pragma once

#include <opencv2/core/core.hpp>
#include "Tensor.hpp"

namespace MaskedCNN {

Tensor<float> loadImage(const std::string &path);
Tensor<float> matToTensor(const cv::Mat &image);
Tensor<float> labelToTensor(const cv::Mat& mask, int label);
cv::Mat maskToMat(const Tensor<float> &tensor);
cv::Mat visualizeOutput(const Tensor<float> &tensor);
Tensor<float> maxarg(const Tensor<float> &data);
Tensor<float> cropLike(const Tensor<float> data, const cv::Mat templateImage, int offset);
cv::Mat cropLike(const cv::Mat data, const cv::Mat templateImage, int offset);
Tensor<float> diffFrames(const cv::Mat frame, const cv::Mat prevFrame, Tensor<float> accumMatrix, int threshold = 0);
cv::Mat saltAndPepper(const cv::Mat frame, double prob);
void addNoiseToVideo(std::string filename, double prob);
cv::Mat median(const cv::Mat frame, int window);
void DenoiseVideo(std::string filename, int window);
}

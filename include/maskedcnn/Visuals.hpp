#pragma once

#include <opencv2/core/core.hpp>
#include "Tensor.hpp"

namespace MaskedCNN {

Tensor<float> loadImage(const std::string &path);
Tensor<float> matToTensor(const cv::Mat &image);
cv::Mat maskToMat(const Tensor<float> &tensor);
cv::Mat visualizeOutput(const Tensor<float> &tensor);
Tensor<float> maxarg(const Tensor<float> &data);
cv::Mat cropLike(const cv::Mat data, const cv::Mat templateImage, int offset);
Tensor<float> diffFrames(const cv::Mat frame, const cv::Mat prevFrame, int threshold = 0);

}

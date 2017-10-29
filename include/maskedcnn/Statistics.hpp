#pragma once

#include <tuple>
#include <utility>
#include "Tensor.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace MaskedCNN {

double IoU(std::vector<std::tuple<int, Tensor<float>, Tensor<float> > > results);
void changePercentOfFrame(cv::Mat currentFrame, int percent);
void changePercentOfFrame(Tensor<float> currentFrame, int percent);

}

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/core.hpp>
#include <vector>
namespace filters
{

typedef std::vector<std::vector<float>> Kernel;

namespace kernels
{

const Kernel GAUSSION_3x3 =
    {
        {0.0625, 0.125, 0.0625},
        {0.125, 0.25, 0.125},
        {0.0625, 0.125, 0.0625}};

const Kernel GAUSSION_5x5 =
    {{0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
     {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
     {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
     {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
     {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};

} // namespace kernels

cv::Mat1f conv2D(const cv::Mat &src, const Kernel &kernel);
cv::Mat1f sobelX(const cv::Mat1b &src);
cv::Mat1f sobelY(const cv::Mat1b &src);
cv::Mat1f sobel(const cv::Mat1b &src);
cv::Mat1b canny(const cv::Mat1b &src, const float lb, const float ub, const int kernel_size = 3);

// cv::Mat1b sobel(const cv::Mat1b &src, int low_thresh, int high_thresh);
// cv::Mat1b calcImageGradientMagnitudeAndDirection(cv::Mat1b src);

} // namespace filters
#endif
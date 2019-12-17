#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/core.hpp>
#include <vector>
namespace filters
{

typedef std::vector<std::vector<float>> Kernel;
const Kernel gaussion(int ksize = 3);
cv::Mat1f conv2D(const cv::Mat &src, const Kernel &kernel);
cv::Mat1f sobelX(const cv::Mat1b &src);
cv::Mat1f sobelY(const cv::Mat1b &src);
cv::Mat1f sobel(const cv::Mat1b &src);
cv::Mat1b canny(const cv::Mat1b &src, const float lb, const float ub, const int kernel_size = 3);

// cv::Mat1b sobel(const cv::Mat1b &src, int low_thresh, int high_thresh);
// cv::Mat1b calcImageGradientMagnitudeAndDirection(cv::Mat1b src);

} // namespace filters
#endif
#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/core.hpp>
#include <vector>
namespace filters
{

typedef std::vector<std::vector<double>> Kernel;
const Kernel gaussion(int ksize = 3); // Create gaussion kernel.
cv::Mat1d conv2D(const cv::Mat &src, const Kernel &kernel);
cv::Mat1d sobelX(const cv::Mat1b &src);
cv::Mat1d sobelY(const cv::Mat1b &src);
cv::Mat1d sobel(const cv::Mat1b &src);
cv::Mat1b canny(const cv::Mat1b &src, const double lb, const double ub, const int kernel_size = 3);

// cv::Mat1b sobel(const cv::Mat1b &src, int low_thresh, int high_thresh);
// cv::Mat1b calcImageGradientMagnitudeAndDirection(cv::Mat1b src);

} // namespace filters
#endif
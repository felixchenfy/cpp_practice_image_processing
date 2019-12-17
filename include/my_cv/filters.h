#ifndef MY_CV_FILTERS_H
#define MY_CV_FILTERS_H

#include <opencv2/core.hpp>
#include <vector>
namespace filters
{

typedef std::vector<std::vector<double>> Kernel;
const Kernel getGaussionKernel(int ksize = 3); // Create gaussion kernel.
cv::Mat1d conv2D(const cv::Mat &src, const Kernel &kernel);
cv::Mat gaussion(const cv::Mat &src, int ksize = 3); // Output same type as input.
cv::Mat1d sobelX(const cv::Mat1b &src);
cv::Mat1d sobelY(const cv::Mat1b &src);
cv::Mat1d sobel(const cv::Mat1b &src);
cv::Mat1b canny(const cv::Mat1b &src, const double lb, const double ub);

} // namespace filters
#endif


#include "my_algos/filters.h"
#include <iostream>

namespace filters
{
// typedef unsigned char uchar;

/**
 * Filter the input gray image with kernel. Output float image.
 */
cv::Mat1f conv2D(const cv::Mat &src, const Kernel &kernel)
{

    assert(src.channels() == 1);
    assert(kernel.size() % 2 == 1 && kernel[0].size() % 2 == 1);

    cv::Mat src_32FC1 = src;
    if (src.depth() != CV_32FC1)
        src.convertTo(src_32FC1, CV_32FC1);

    const int r1 = kernel.size() / 2;    // Radius of the kernel.
    const int r2 = kernel[0].size() / 2; // Radius of the kernel.

    cv::Mat1f dst = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);
    for (int i = r1; i < src.rows - r1; ++i)
        for (int j = r2; j < src.cols - r2; ++j)
        {
            float sums = 0;
            for (int m = -r1; m <= r1; m++)
                for (int n = -r2; n <= r2; n++)
                    sums += src_32FC1.at<float>(i + m, j + n) * kernel[m + r1][n + r2];
            dst.at<float>(i, j) = sums;
        }
    return dst;
}

cv::Mat1f sobelX(const cv::Mat1b &src)
{
    // https://en.wikipedia.org/wiki/Sobel_operator
    const Kernel kernel = {{-1., 0., 1.},
                           {-2., 0., 2.},
                           {-1., 0., 1.}};
    const Kernel sub_kernel_1 = {{-1., 0., 1.}};                     // Horizontal.
    const Kernel sub_kernel_2 = {{1.}, {2.}, {1.}};                  // Vertical.
    cv::Mat1f dst = conv2D(conv2D(src, sub_kernel_1), sub_kernel_2); // Faster.
    // cv::Mat1f dst = conv2D(src, kernel);
    return dst;
}

cv::Mat1f sobelY(const cv::Mat1b &src)
{
    const Kernel kernel = {{-1., -2., -1.},
                           {0., 0., 0.},
                           {1., 2., 1.}};
    // https://en.wikipedia.org/wiki/Sobel_operator
    const Kernel sub_kernel_1 = {{1., 2., 1.}};      // Horizontal.
    const Kernel sub_kernel_2 = {{-1.}, {0.}, {1.}}; // Vertical.
    cv::Mat1f dst = conv2D(conv2D(src, sub_kernel_1), sub_kernel_2); // Faster.
    // cv::Mat1f dst = conv2D(src, kernel);
    return dst;
}

cv::Mat1f sobel(const cv::Mat1b &src)
{
    cv::Mat1f sobel_x = sobelX(src);
    cv::Mat1f sobel_y = sobelY(src);
    cv::Mat1f dst_img_grad_mag = cv::Mat::zeros(src.size(), CV_32FC1);
    for (int i = 0; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j)
            dst_img_grad_mag.at<float>(i, j) = sqrt(
                pow(sobel_x.at<float>(i, j), 2.) +
                pow(sobel_y.at<float>(i, j), 2.));
    return dst_img_grad_mag;
}

// /**
//  * Compute sobel gradient.
//  */
// cv::Mat1b sobel(const cv::Mat1b &src, int low_thresh, int high_thresh){
//     const int KERNEL_X[] = {-1,     }
} // namespace filters

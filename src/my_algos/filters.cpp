

#include "my_algos/filters.h"

namespace filters
{
// typedef unsigned char uchar;

/**
 * Filter the input gray image with kernel.
 */
cv::Mat1b conv2D(const cv::Mat1b &src, const Kernel &kernel)
{
    assert(src.depth() == CV_8U);
    assert(src.channels() == 1);
    assert(kernel.size() % 2 == 1 && kernel[0].size() % 2 == 1);

    const int r1 = kernel.size() / 2;    // Radius of the kernel.
    const int r2 = kernel[0].size() / 2; // Radius of the kernel.

    // Init dst image.
    const int n_rows = src.rows;
    const int n_cols = src.cols;
    cv::Mat1b dst = cv::Mat::zeros(n_rows, n_cols, CV_8U);

    // -- Convolute through the image.
    // -- Case 1: 3x3, 5x5, ...
    if (r1 > 0 && r2 > 0)
    {
        for (int i = r1; i < n_rows - r1; ++i)
            for (int j = r2; j < n_cols - r2; ++j)
            {
                float sums = 0;
                for (int m = -r1; m <= r1; m++)
                    for (int n = -r2; n <= r2; n++)
                        sums += static_cast<float>(src.at<uchar>(i + m, j + n)) * kernel[m + r1][n + r2];
                dst.at<uchar>(i, j) = static_cast<uchar>(sums);
            }
    }
    else if (r1 > 0)
    {
        for (int i = r1; i < n_rows - r1; ++i)
            for (int j = r2; j < n_cols - r2; ++j)
            {
                float sums = 0;
                for (int m = -r1; m <= r1; m++)
                    sums += static_cast<float>(src.at<uchar>(i + m, j)) * kernel[m + r1][0];
                dst.at<uchar>(i, j) = static_cast<uchar>(sums);
            }
    }
    else if (r2 > 0)
    {
        for (int i = r1; i < n_rows - r1; ++i)
            for (int j = r2; j < n_cols - r2; ++j)
            {
                float sums = 0;
                for (int n = -r2; n <= r2; n++)
                    sums += static_cast<float>(src.at<uchar>(i, j + n)) * kernel[0][n + r2];
                dst.at<uchar>(i, j) = static_cast<uchar>(sums);
            }
    }
    else
    {
        assert(0);
    }
    return dst;
}

cv::Mat1b sobelX(const cv::Mat1b &src)
{
    // https://en.wikipedia.org/wiki/Sobel_operator
    const Kernel kernel_1 = {{-1., 0., 1.}};    // Horizontal.
    const Kernel kernel_2 = {{1.}, {2.}, {1.}}; // Vertical.
    cv::Mat1b dst = conv2D(src, kernel_1);
    dst = conv2D(dst, kernel_2);
    return dst;
}

cv::Mat1b sobelY(const cv::Mat1b &src)
{
    // https://en.wikipedia.org/wiki/Sobel_operator
    const Kernel kernel_1 = {{1., 2., 1.}};      // Horizontal.
    const Kernel kernel_2 = {{1.}, {0.}, {-1.}}; // Vertical.
    cv::Mat1b dst = conv2D(src, kernel_1);
    dst = conv2D(dst, kernel_2);
    return dst;
}

cv::Mat1f sobel(const cv::Mat1b &src, cv::Mat1f *dst_img_grad_direction)
{
    constexpr float SCALE_MAGNITUDE = 1.0;
    cv::Mat1b sobel_x = sobelX(src);
    cv::Mat1b sobel_y = sobelY(src);
    cv::Mat1f dst_img_grad_mag = cv::Mat::zeros(src.size(), CV_32FC1);
    float *p_dst;
    uchar *p_sobel_x, *p_sobel_y;
    for (int i = 0; i < src.rows; ++i)
    {
        p_sobel_x = sobel_x.ptr<uchar>(i);
        p_sobel_y = sobel_y.ptr<uchar>(i);
        p_dst = dst_img_grad_mag.ptr<float>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            p_dst[j] = sqrt(pow(p_sobel_x[j], 2.) + pow(p_sobel_y[j], 2.)) *
                       SCALE_MAGNITUDE;
        }
    }
    return dst_img_grad_mag;
}

// /**
//  * Compute sobel gradient.
//  */
// cv::Mat1b sobel(const cv::Mat1b &src, int low_thresh, int high_thresh){
//     const int KERNEL_X[] = {-1,     }
} // namespace filters

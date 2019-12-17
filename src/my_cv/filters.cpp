

#include "my_cv/filters.h"
#include "my_cv/cv_basics.h"
#include <iostream>
#include <cmath>
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

    cv::Mat1f dst = cv::Mat::zeros(src.rows, src.cols, CV_32FC1); // Pad zeros.
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
    const Kernel sub_kernel_1 = {{-1., 0., 1.}};    // Horizontal.
    const Kernel sub_kernel_2 = {{1.}, {2.}, {1.}}; // Vertical.
    // cv::Mat1f dst = conv2D(conv2D(src, sub_kernel_1), sub_kernel_2); // Faster.
    cv::Mat1f dst = conv2D(src, kernel);
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
    // cv::Mat1f dst = conv2D(conv2D(src, sub_kernel_1), sub_kernel_2); // Faster.
    cv::Mat1f dst = conv2D(src, kernel);
    return dst;
}

cv::Mat1f sobel(const cv::Mat1b &src)
{
    cv::Mat1f sobel_x = sobelX(src);
    cv::Mat1f sobel_y = sobelY(src);
    cv::Mat1f dst_Ig = cv::Mat::zeros(src.size(), CV_32FC1);
    for (int i = 0; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j)
            dst_Ig.at<float>(i, j) = sqrt(
                pow(sobel_x.at<float>(i, j), 2.) +
                pow(sobel_y.at<float>(i, j), 2.));
    return dst_Ig;
}

void _cannyDfs(
    const cv::Mat1f &Ig,
    const std::vector<cv::Point2i> &neighbors,
    cv::Mat1b *visited,
    cv::Mat1b *res_mask,
    const int i, const int j, const float lb)
{
    if (i < 0 || i >= Ig.rows || j < 0 || j >= Ig.cols)
        return; // Invalid pixel.
    if (visited->at<uchar>(i, j) != 0)
        return; // Already visited before.
    if (Ig.at<float>(i, j) >= lb)
    {
        res_mask->at<uchar>(i, j) = 255;
        visited->at<uchar>(i, j) = 255;

        // DFS the neighboring pixels.
        for (const cv::Point2i &neighbor : neighbors)
        {
            _cannyDfs(Ig, neighbors, visited, res_mask,
                      i + neighbor.y,
                      j + neighbor.x,
                      lb);
        }
    }
}

/**
 * Canny edge detection.
 * Algorithm: https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html
 * Procedures:
 *      1. Compute image gradient Ix, Iy.
 *      2. Compute gradient's magnitude Ig and direction Id.
 *      3. Non maximum suppression along the gradient direction and update Ig.
 *      4. Mask pixels (i, j) as edge 
 *          if: Ig[i, j] >= ub,
 *          or: Ig[i, j] >= lb and is connected to a pixel >= ub.
 */
cv::Mat1b canny(const cv::Mat1b &src, const float lb, const float ub, const int kernel_size)
{
    // -- Check input.
    assert(kernel_size == 3); // Only support 3.

    // Scale the gradient's magnitude to match with OpenCV. This is set by my experiment.
    constexpr float SCALE_GRADIENT_MAG = 1.75;

    // -- Step 1:  Compute image gradient Ix, Iy.
    // Blur image.
    const int r1 = 1; // radius of the gaussian filter.
    cv::Mat src_blurred = conv2D(src, kernels::GAUSSION_3x3);
    src_blurred.convertTo(src_blurred, CV_8UC1); // float to uchar.

    // Sobel gradient.
    const int r2 = 1;                   // radius of the sobel kernel.
    cv::Mat1f Ix = sobelX(src_blurred); // Gradient x.
    cv::Mat1f Iy = sobelY(src_blurred); // Gradient y.

    // -- Step 2: Compute gradient's magnitude Ig and direction Id.
    cv::Mat1f Ig = cv::Mat::zeros(src.size(), CV_32FC1); // Maginitude.
    cv::Mat1f Id = cv::Mat::zeros(src.size(), CV_32FC1); // Direction.
    const int r = r1 + r2;                               // Total offset.
    for (int i = r; i < src.rows - r; ++i)
        for (int j = r; j < src.cols - r; ++j)
        {
            float dx = Ix.at<float>(i, j);
            float dy = Iy.at<float>(i, j);
            // Ig.at<float>(i, j) = sqrt(pow(dx, 2.) + pow(dy, 2.)) * SCALE_GRADIENT_MAG;
            Ig.at<float>(i, j) = (fabs(dx) + fabs(dy)) * SCALE_GRADIENT_MAG;
            Id.at<float>(i, j) = atan2(dy, dx);
        }
    // cv::imwrite("canny_gradient_mag.png", cv_basics::float2uint8(Ig / 4.0));
    // cv::imwrite("canny_gradient_dir.png", cv_basics::float2uint8(Id / M_PI * 127.0 + 127.0));

    // -- Step 3: Non maximum suppression along the gradient direction.
    cv::Mat1f Ig_tmp = Ig.clone();
    const std::vector<cv::Point2i> neighbors = {
        {-1, 00}, // -180 degrees.
        {-1, -1},
        {00, -1},
        {+1, -1},
        {+1, 00},
        {+1, +1},
        {00, +1},
        {-1, +1}, // +180 degrees.
    };
    for (int i = r; i < src.rows - r; ++i)
        for (int j = r; j < src.cols - r; ++j)
        {
            int index = lround(Id.at<float>(i, j) / M_PI_4) + 4; // 0~7
            index = index == 8 ? 0 : index;                      // 8 means 180 degrees, which is the same as -180 degrees.
            int d_row = neighbors.at(index).y, d_col = neighbors.at(index).x;
            int r1 = i + d_row, r2 = i - d_row;
            int c1 = j + d_col, c2 = j - d_col;
            if (Ig.at<float>(i, j) < Ig.at<float>(r1, c1) ||
                Ig.at<float>(i, j) < Ig.at<float>(r2, c2))
                // Should be "<", to deal with a row of same gradient.
                Ig_tmp.at<float>(i, j) = 0.;
        }
    Ig = Ig_tmp;

    // -- Mask pixels (i, j) as edge if Ig[i, j] >= ub.
    cv::Mat1b res_mask = cv::Mat::zeros(src.size(), CV_8U);
    cv::Mat1b visited = cv::Mat::zeros(src.size(), CV_8U);
    for (int i = 0; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j)
        {
            if (Ig.at<float>(i, j) >= ub)
                _cannyDfs(Ig, neighbors, &visited, &res_mask, i, j, lb);
        }

    return res_mask;
}

// /**
//  * Compute sobel gradient.
//  */
// cv::Mat1b sobel(const cv::Mat1b &src, int low_thresh, int high_thresh){
//     const int KERNEL_X[] = {-1,     }
} // namespace filters

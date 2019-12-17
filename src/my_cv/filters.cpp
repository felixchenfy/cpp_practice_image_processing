

#include "my_cv/filters.h"
#include "my_cv/cv_basics.h"
#include <iostream>
#include <cmath>
namespace filters
{

/**
 * Create gaussion kernel.
 */
inline const Kernel gaussion(int ksize)
{
    assert(ksize % 2 == 1 && ksize >= 3 && ksize <= 101);
    // if (ksize == 3)
    // {
    //     static const Kernel GAUSSION_3x3 =
    //         {
    //             {0.0625, 0.125, 0.0625},
    //             {0.125, 0.25, 0.125},
    //             {0.0625, 0.125, 0.0625}};
    //     return GAUSSION_3x3;
    // }
    // else if (ksize == 5)
    // {

    //     static const Kernel GAUSSION_5x5 =
    //         {{0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
    //          {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
    //          {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
    //          {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
    //          {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};
    //     return GAUSSION_5x5;
    // }
    // else
    // {
    // How to determine sigma: https://blog.shinelee.me/2018/09-19-%E5%A6%82%E4%BD%95%E7%A1%AE%E5%AE%9A%E9%AB%98%E6%96%AF%E6%BB%A4%E6%B3%A2%E7%9A%84%E6%A0%87%E5%87%86%E5%B7%AE%E5%92%8C%E7%AA%97%E5%8F%A3%E5%A4%A7%E5%B0%8F.html
    double sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
    std::vector<std::vector<double>> kernel(ksize, std::vector<double>(ksize, 0.));
    double c = static_cast<double>(ksize / 2);
    double sums = 0;
    for (int i = 0; i < ksize; i++)
        for (int j = 0; j < ksize; j++)
        {
            kernel[i][j] = exp(-(pow(i - c, 2.) + pow(j - c, 2)) / 0.5 / pow(sigma, 2.));
            sums += kernel[i][j];
        }
    for (int i = 0; i < ksize; i++)
        for (int j = 0; j < ksize; j++)
            kernel[i][j] /= sums;
    return kernel;
    // }
}

/**
 * Filter the input gray image with kernel. Output double image.
 */
cv::Mat1d conv2D(const cv::Mat &src, const Kernel &kernel)
{

    assert(src.channels() == 1);
    assert(kernel.size() % 2 == 1 && kernel[0].size() % 2 == 1);
    assert(kernel.size() <= src.rows);
    assert(kernel[0].size() <= src.cols);

    // Convert data type to 64FC1.
    cv::Mat src_64FC1 = src;
    if (src.depth() != CV_64FC1)
        src.convertTo(src_64FC1, CV_64FC1);

    // Radius of the kernel.
    const int r1 = kernel.size() / 2;
    const int r2 = kernel[0].size() / 2;

    // Convolute the image.
    cv::Mat1d dst = cv::Mat::zeros(src.rows, src.cols, CV_64FC1); // Pad zeros.
    for (int i = r1; i < src.rows - r1; ++i)
        for (int j = r2; j < src.cols - r2; ++j)
        {
            double sums = 0;
            for (int m = -r1; m <= r1; m++)
                for (int n = -r2; n <= r2; n++)
                    sums += src_64FC1.at<double>(i + m, j + n) * kernel[m + r1][n + r2];
            dst.at<double>(i, j) = sums;
        }
    return dst;
}

cv::Mat1d sobelX(const cv::Mat1b &src)
{
    // https://en.wikipedia.org/wiki/Sobel_operator
    const Kernel kernel = {{-1., 0., 1.},
                           {-2., 0., 2.},
                           {-1., 0., 1.}};
    const Kernel sub_kernel_1 = {{-1., 0., 1.}};    // Horizontal.
    const Kernel sub_kernel_2 = {{1.}, {2.}, {1.}}; // Vertical.
    // cv::Mat1d dst = conv2D(conv2D(src, sub_kernel_1), sub_kernel_2); // Faster.
    cv::Mat1d dst = conv2D(src, kernel);
    return dst;
}

cv::Mat1d sobelY(const cv::Mat1b &src)
{
    const Kernel kernel = {{-1., -2., -1.},
                           {0., 0., 0.},
                           {1., 2., 1.}};
    // https://en.wikipedia.org/wiki/Sobel_operator
    const Kernel sub_kernel_1 = {{1., 2., 1.}};      // Horizontal.
    const Kernel sub_kernel_2 = {{-1.}, {0.}, {1.}}; // Vertical.
    // cv::Mat1d dst = conv2D(conv2D(src, sub_kernel_1), sub_kernel_2); // Faster.
    cv::Mat1d dst = conv2D(src, kernel);
    return dst;
}

cv::Mat1d sobel(const cv::Mat1b &src)
{
    cv::Mat1d sobel_x = sobelX(src);
    cv::Mat1d sobel_y = sobelY(src);
    cv::Mat1d dst_Ig = cv::Mat::zeros(src.size(), CV_64FC1);
    for (int i = 0; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j)
            dst_Ig.at<double>(i, j) = sqrt(
                pow(sobel_x.at<double>(i, j), 2.) +
                pow(sobel_y.at<double>(i, j), 2.));
    return dst_Ig;
}

void _cannyDfs(
    const cv::Mat1d &Ig,
    const std::vector<cv::Point2i> &neighbors,
    cv::Mat1b *visited,
    cv::Mat1b *res_mask,
    const int i, const int j, const double lb)
{
    if (i < 0 || i >= Ig.rows || j < 0 || j >= Ig.cols)
        return; // Invalid pixel.
    if (visited->at<uchar>(i, j) != 0)
        return; // Already visited before.
    if (Ig.at<double>(i, j) >= lb)
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
cv::Mat1b canny(const cv::Mat1b &src, const double lb, const double ub, const int kernel_size)
{
    // -- Check input.
    assert(kernel_size == 3); // Only support 3.

    // Scale the gradient's magnitude to match with OpenCV. This is set by my experiment.
    constexpr double SCALE_GRADIENT_MAG = 1.75;

    // -- Step 1:  Compute image gradient Ix, Iy.
    // Blur image.
    const int r1 = 1; // radius of the gaussian filter.
    cv::Mat src_blurred = conv2D(src, gaussion(3));
    src_blurred.convertTo(src_blurred, CV_8UC1); // double to uchar.

    // Sobel gradient.
    const int r2 = 1;                   // radius of the sobel kernel.
    cv::Mat1d Ix = sobelX(src_blurred); // Gradient x.
    cv::Mat1d Iy = sobelY(src_blurred); // Gradient y.

    // -- Step 2: Compute gradient's magnitude Ig and direction Id.
    cv::Mat1d Ig = cv::Mat::zeros(src.size(), CV_64FC1); // Maginitude.
    cv::Mat1d Id = cv::Mat::zeros(src.size(), CV_64FC1); // Direction.
    const int r = r1 + r2;                               // Total offset.
    for (int i = r; i < src.rows - r; ++i)
        for (int j = r; j < src.cols - r; ++j)
        {
            double dx = Ix.at<double>(i, j);
            double dy = Iy.at<double>(i, j);
            // Ig.at<double>(i, j) = sqrt(pow(dx, 2.) + pow(dy, 2.)) * SCALE_GRADIENT_MAG;
            Ig.at<double>(i, j) = (fabs(dx) + fabs(dy)) * SCALE_GRADIENT_MAG;
            Id.at<double>(i, j) = atan2(dy, dx);
        }
    // cv::imwrite("canny_gradient_mag.png", cv_basics::double2uint8(Ig / 4.0));
    // cv::imwrite("canny_gradient_dir.png", cv_basics::double2uint8(Id / M_PI * 127.0 + 127.0));

    // -- Step 3: Non maximum suppression along the gradient direction.
    cv::Mat1d Ig_tmp = Ig.clone();
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
            int index = lround(Id.at<double>(i, j) / M_PI_4) + 4; // 0~7
            index = index == 8 ? 0 : index;                       // 8 means 180 degrees, which is the same as -180 degrees.
            int d_row = neighbors.at(index).y, d_col = neighbors.at(index).x;
            int r1 = i + d_row, r2 = i - d_row;
            int c1 = j + d_col, c2 = j - d_col;
            if (Ig.at<double>(i, j) < Ig.at<double>(r1, c1) ||
                Ig.at<double>(i, j) < Ig.at<double>(r2, c2))
                // Should be "<", to deal with a row of same gradient.
                Ig_tmp.at<double>(i, j) = 0.;
        }
    Ig = Ig_tmp;

    // -- Mask pixels (i, j) as edge if Ig[i, j] >= ub.
    cv::Mat1b res_mask = cv::Mat::zeros(src.size(), CV_8U);
    cv::Mat1b visited = cv::Mat::zeros(src.size(), CV_8U);
    for (int i = 0; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j)
        {
            if (Ig.at<double>(i, j) >= ub)
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

#ifndef MY_CV_COMMONS_H
#define MY_CV_COMMONS_H

/**
 * Functions:
 *      readImage
 *      gray2color
 *      any2color
 *      float2uint8
 *      display_images
 *      isLocalMax
 *      setNeighborsToZero
 *      setBlockToZero
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <vector>
#include <iostream>

namespace cv_commons
{

/**
 * Read a color image from disk.
 * If image path is invalid, print the filename and then exit the program.
 */
inline cv::Mat readImage(const std::string &filename = "")
{
    cv::Mat src = imread(cv::samples::findFile(filename), cv::IMREAD_COLOR); // Load an image
    if (src.empty())
    {
        std::cout << "Could not open or find the image: \n"
                  << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    return src;
}

/**
 * Gray image to color image by cv::cvtColor.
 */
inline cv::Mat3b gray2color(const cv::Mat1b &gray)
{
    assert(gray.channels() == 1);
    cv::Mat3b color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    return color;
}

// Convert any image to 8UC3 color image.
inline cv::Mat3b any2color(const cv::Mat &image)
{
    assert(image.channels() == 1 || image.channels() == 3);
    cv::Mat I_uint8;
    image.convertTo(I_uint8, CV_8U); // To uint8.
    if (I_uint8.channels() == 1)
        cv::cvtColor(I_uint8, I_uint8, cv::COLOR_GRAY2BGR);
    return I_uint8;
}

/**
 * Convert image from float/double to uchar by:
 *      ((abs(gray) * scale) + inc_value).to_uint8()
 * The channel number is not changed.
 */
inline cv::Mat float2uint8(
    const cv::Mat &gray,
    bool take_abs = false,
    double scale = 1.0,
    double inc_value = 0.)
{
    cv::Mat tmp = take_abs ? cv::abs(gray) : gray; // Take abs.
    tmp = tmp * scale + inc_value;                 // Scale and shift value.
    cv::Mat dst;
    tmp.convertTo(dst, CV_8U); // To uint8.
    return dst;
}

/**
 * Display a vector of images.
 * The image can be uint8/float/double, 1 or 3 channels.
 */
inline void display_images(
    const std::vector<cv::Mat> images,
    const std::string WINDOW_NAME = "Images",
    const int wait_key_ms = 0)
{
    cv::Mat img0 = images[0];
    const int rows = img0.rows, cols = img0.cols;
    cv::Mat img_disp = any2color(img0);
    for (int i = 1; i < images.size(); i++)
    {
        const cv::Mat &image = images[i];
        assert(image.rows == rows && image.cols == cols);
        cv::Mat next_image = any2color(image);
        cv::hconcat(img_disp, next_image, img_disp);
    }
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::imshow(WINDOW_NAME, img_disp);
    if (wait_key_ms >= 0)
        cv::waitKey(wait_key_ms);
    return;
}

/**
 * Check if img[row, col] is >= {all its neighbor elements}.
 * @param radius Radius of the neighbor square.
 */
template <typename pixel_type>
bool isLocalMax(const cv::Mat &img, int row, int col, int radius)
{
    pixel_type center_value = img.at<pixel_type>(row, col);
    for (int m = -radius; m <= radius; m++)
        for (int n = -radius; n <= radius; n++)
        {
            if (m == 0 || n == 0)
                continue;
            int r = row + m, c = col + n;
            if (r < 0 || c < 0 || r >= img.rows || c >= img.cols)
                continue;
            if (img.at<pixel_type>(r, c) > center_value)
                return false;
        }
    return true;
}

/**
 * Set neighboring pixels of img[row, col] to zero.
 * @param radius Radius of the neighbor.
 */
template <typename pixel_type>
void setNeighborsToZero(cv::Mat *img, int row, int col, int radius)
{
    assert(img != nullptr);
    for (int m = -radius; m <= radius; m++)
        for (int n = -radius; n <= radius; n++)
        {
            if (m == 0 || n == 0)
                continue;
            int r = row + m, c = col + n;
            if (r < 0 || c < 0 || r >= img->rows || c >= img->cols)
                continue;
            img->at<pixel_type>(r, c) = static_cast<pixel_type>(0);
        }
}

/**
 * Set a block of pixels around img[row, col] to zero.
 * (The only difference to setNeighborsToZero() is that 
 *  the center pixel is also set as zero.)
 * @param radius Radius of the block.
 */
template <typename pixel_type>
void setBlockToZero(cv::Mat *img, int row, int col, int radius)
{
    assert(img != nullptr);
    for (int m = -radius; m <= radius; m++)
        for (int n = -radius; n <= radius; n++)
        {
            int r = row + m, c = col + n;
            if (r < 0 || c < 0 || r >= img->rows || c >= img->cols)
                continue;
            img->at<pixel_type>(r, c) = static_cast<pixel_type>(0);
        }
}

} // namespace cv_commons
#endif
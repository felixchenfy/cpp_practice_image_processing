#ifndef BASICS_H
#define BASICS_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <vector>

namespace cv_basics
{

/**
 * Gray image to color image.
 */
inline cv::Mat3b gray2color(const cv::Mat1b &gray)
{
    assert(gray.channels() == 1);
    cv::Mat3b color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    return color;
}

/**
 * Float image to uchar image.
 * (( abs(gray) * scale) + inc_value).to_uint8()
 */
inline cv::Mat1b float2uint8(
    const cv::Mat1f &gray,
    bool take_abs = false,
    float scale = 1.0,
    float inc_value = 0.)
{
    assert(gray.channels() == 1);
    cv::Mat1b dst;
    cv::Mat1f tmp = take_abs ? cv::abs(gray) : gray;
    tmp = tmp * scale + inc_value; // Change value.
    tmp.convertTo(dst, CV_8UC1);   // To uint8.
    return dst;
}

/**
 * Display a vector of uchar or float images.
 */
inline void display_images(const std::vector<cv::Mat> images, const std::string WINDOW_NAME = "Images")
{
    cv::Mat img0 = images[0];
    const int rows = img0.rows, cols = img0.cols;
    cv::Mat img_disp = img0.channels() == 3 ? img0 : gray2color(img0);
    for (int i = 1; i < images.size(); i++)
    {
        const cv::Mat &image = images[i];
        assert(image.rows == rows && image.cols == cols);
        cv::Mat next_image = image.channels() == 3 ? image : gray2color(image);
        cv::hconcat(img_disp, next_image, img_disp);
    }
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::imshow(WINDOW_NAME, img_disp);
    cv::waitKey(0);
    return;
}

/**
 * Check if img[row, col] is >= all its neighbor elements.
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
 * (The only difference to setNeighborsToZero() is that the center pixel is also set as zero.)
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

} // namespace cv_basics
#endif

#include "my_algos/filters.h"
#include "my_algos/basics.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

using namespace std;

typedef vector<vector<float>> Kernel;

bool assert_kernel_sums_to_one(const Kernel &kernel, const float EPS = 0.0001)
{
    int r = kernel.size(), c = kernel[0].size();
    float sums = 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            sums += kernel[i][j];
    assert(abs(1.0 - sums) <= EPS);
}

cv::Mat readImage(const string &filename = "")
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

// ---------------------------------- Main -------------------------------------- //

void test_kernels_values()
{
    //  Test whether Gaussian kernels sum to one.
    assert_kernel_sums_to_one(filters::kernels::GAUSSION_3x3);
    assert_kernel_sums_to_one(filters::kernels::GAUSSION_5x5);
}

void test_conv2d()
{

    // -- Read image
    // const string filename = "data/color_chessboard.jpg";
    const string filename = "data/simple_shapes2.png";
    cv::Mat src = readImage(filename);
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
    std::cout << "Image size: " << src.size() << std::endl; // [cols, rows] = [320, 240]

    // -- Filter
    cv::Mat blurred;
    blurred = filters::conv2D(src_gray, filters::kernels::GAUSSION_5x5);
    blurred = filters::conv2D(blurred, filters::kernels::GAUSSION_5x5);

    // -- Show image
    const std::string WINDOW_NAME = "Color/Gray/Blurred";
    basics::display_images({src, src_gray, blurred}, WINDOW_NAME);
}

void test_sobel()
{

    // -- Read image
    // const string filename = "data/color_chessboard.jpg";
    const string filename = "data/simple_shapes2.png";
    cv::Mat src_gray;
    cv::cvtColor(readImage(filename), src_gray, cv::COLOR_BGR2GRAY);

    // -- Filter
    cv::Mat1f edge_x = filters::sobelX(src_gray);
    cv::Mat1f edge_y = filters::sobelY(src_gray);
    cv::Mat1f edge = filters::sobel(src_gray);

    // -- Float 2 uint8 for display.
    constexpr bool TAKE_ABS = true;
    constexpr float SCALE_GRAD = 0.3, INC_GRAD = 0;
    cv::Mat1b disp_edge_x = basics::float2uint8(edge_x, TAKE_ABS, SCALE_GRAD, INC_GRAD);
    cv::Mat1b disp_edge_y = basics::float2uint8(edge_y, TAKE_ABS, SCALE_GRAD, INC_GRAD);
    cv::Mat1b disp_edge = basics::float2uint8(edge, TAKE_ABS, SCALE_GRAD, INC_GRAD);

    // -- Show image
    const std::string WINDOW_NAME = "Original/SobelX/SobelY/Sobel";
    basics::display_images(
        {src_gray, disp_edge_x, disp_edge_y, disp_edge},
        WINDOW_NAME);
}

int main(int argc, char const *argv[])
{
    // test_conv2d();
    test_sobel();
}

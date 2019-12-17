
#include "my_cv/filters.h"
#include "my_cv/cv_basics.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

typedef std::vector<std::vector<double>> Kernel;

bool assert_kernel_sums_to_one(const Kernel &kernel, const double EPS = 0.0001)
{
    int r = kernel.size(), c = kernel[0].size();
    double sums = 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            sums += kernel[i][j];
    assert(abs(1.0 - sums) <= EPS);
}

cv::Mat readImage(const std::string &filename = "")
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

void test_kernel_sums_to_one()
{
    //  Test whether Gaussian kernels sum to one.
    assert_kernel_sums_to_one(filters::gaussion(3));
    assert_kernel_sums_to_one(filters::gaussion(5));
    assert_kernel_sums_to_one(filters::gaussion(7));
}

void test_sobel_and_gaussion()
{

    // -- Read image
    const std::string filename = "data/color_chessboard.jpg";
    // const std::string filename = "data/simple_shapes2.png";
    cv::Mat src_gray;
    cv::cvtColor(readImage(filename), src_gray, cv::COLOR_BGR2GRAY);

    // -- Filter
    cv::Mat1f edge_x = filters::sobelX(src_gray);
    cv::Mat1f edge_y = filters::sobelY(src_gray);
    cv::Mat1f edge = filters::sobel(src_gray);
    cv::Mat1f gaussion = filters::conv2D(src_gray, filters::gaussion(21));

    // -- Convert image from double to uint8 for display.
    constexpr bool TAKE_ABS = true;
    constexpr double SCALE_GRAD = 0.3;
    cv::Mat1b disp_edge_x = cv_basics::double2uint8(edge_x, TAKE_ABS, SCALE_GRAD);
    cv::Mat1b disp_edge_y = cv_basics::double2uint8(edge_y, TAKE_ABS, SCALE_GRAD);
    cv::Mat1b disp_edge = cv_basics::double2uint8(edge, TAKE_ABS, SCALE_GRAD);
    cv::Mat1b disp_gaussion = cv_basics::double2uint8(gaussion);

    // -- Show image.
    const std::string WINDOW_NAME =
        "Original     /     Gaussion     /     SobelX     /     SobelY     /     Sobel";
    cv_basics::display_images(
        {src_gray, disp_gaussion, disp_edge_x, disp_edge_y, disp_edge},
        WINDOW_NAME);
}

int main(int argc, char const *argv[])
{
    test_kernel_sums_to_one();
    test_sobel_and_gaussion();
}

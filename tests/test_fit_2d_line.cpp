
#include "my_ransac/models.h"
#include "my_cv/cv_commons.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

cv::Mat create_white_image(
    const int row, const int col)
{
    return cv::Mat::zeros(row, col, CV_8UC3) + cv::Scalar(255, 255, 255);
}

void draw_points(
    cv::Mat *img_disp,
    const std::vector<cv::Point2d> points)
{

    constexpr int RADIUS = 10;
    constexpr int LINE_TYPE = 8;
    const cv::Scalar COLOR_BLUE{255, 0, 0};
    for (const auto p : points)
        cv::circle(*img_disp, {int(p.x), int(p.y)},
                   RADIUS, COLOR_BLUE, CV_FILLED, LINE_TYPE);
}

cv::Mat fit_points(const std::vector<cv::Point2d> &points)
{

    models::ModelLine2D model;
    model.train(points);
    model.printParam();
    std::vector<double> abc = model.getParam();

    cv::Mat img_disp = create_white_image(320, 480);
    model.draw(&img_disp, {0, 0, 255}, 3); // Draw the fitted line.
    draw_points(&img_disp, points);        // Draw points.

    return img_disp;
}

cv::Mat test_fit_2_points()
{
    const std::vector<cv::Point2d> points = {
        {100, 100},
        {100, 200}};
    return fit_points(points);
}

cv::Mat test_fit_5_points()
{
    const std::vector<cv::Point2d> points = {
        {100 + 10, 100 + 30},
        {150 - 20, 150 - 10},
        {200 + 30, 200 - 01},
        {250 - 10, 250 + 04},
        {300 + 05, 300 - 24}};
    return fit_points(points);
}

int main(int argc, char const *argv[])
{
    cv::Mat img_disp1 = test_fit_2_points();
    cv::Mat img_disp2 = test_fit_5_points();

    cv_commons::display_images(
        {img_disp1, img_disp2},
        "Test1: Fit 2 points."
        "                                                       "
        "Test2: Fit 5 points.");
    return 0;
}

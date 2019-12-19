
#include "my_ransac/model_2d_line.h"
#include "my_ransac/ransac.h"

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

cv::Mat fit_points(const std::vector<cv::Point2d> &points, const int n_inital_samples)
{

    // Call RANSAC.
    models::Model2dLine model;
    typedef cv::Point2d Datum;         // (x, y)
    typedef std::vector<double> Param; // (a, b, c). Line eq: ax+by+c=0.

    const int iterations = 20;
    const int min_pts_as_good_fit = 3;
    const int dist_thresh_for_inlier_point = 30.0;
    const bool is_print = true;
    std::vector<unsigned int> indices = ransac::ransac<Datum, Param>(
        points, &model, n_inital_samples,
        iterations, min_pts_as_good_fit, dist_thresh_for_inlier_point, is_print);

    if (indices.empty())
        throw std::runtime_error("Failed to fit line by RANSAC.");

    // Print.
    std::cout << "Inliner points: ";
    for (unsigned int idx : indices)
        std::cout << idx << ", ";
    std::cout << std::endl;
    model.printParam();

    // Draw the fitted line.
    cv::Mat img_disp = create_white_image(320, 480);
    model.draw(&img_disp, {0, 0, 255}, 3); // Draw the fitted line.
    draw_points(&img_disp, points);        // Draw points.

    return img_disp;
}

void test_ransac_fit_2d_line()
{
    const std::vector<cv::Point2d> points = {
        // Inlier.
        {100 + 10, 100 - 20},
        {150 - 20, 150 - 10},
        {200 + 20, 200 - 01},
        {250 - 10, 250 + 04},
        {330 + 05, 300 - 24},
        // Outlier.
        {100, 300},
        {300, 100},
    };

    int n_inital_samples1 = 2;
    cv::Mat img_disp1 = fit_points(points, n_inital_samples1);

    int n_inital_samples2 = 4;
    cv::Mat img_disp2 = fit_points(points, n_inital_samples2);

    cv_commons::display_images(
        {img_disp1, img_disp2},
        "n_inital_samples1=2"
        "                                                          "
        "n_inital_samples1=4");
    return;
}

int main(int argc, char const *argv[])
{
    test_ransac_fit_2d_line();
    return 0;
}

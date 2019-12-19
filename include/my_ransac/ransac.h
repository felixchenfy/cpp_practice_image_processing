#ifndef MY_RANSAC_RANSAC_H
#define MY_RANSAC_RANSAC_H

#include "my_ransac/abstract_model.hpp"
#include "my_ransac/random_data_sampler.hpp"

#include <iostream>
#include <vector>

namespace ransac
{
typedef std::vector<unsigned int> Indices;

unsigned int countDataSmallerThan(const std::vector<double> &errors, const double thresh)
{
    unsigned int cnt = 0;
    for (const double error : errors)
        if (error < thresh)
            cnt += 1;
    return cnt;
}

Indices getIndicesOfDataSmallerThan(const std::vector<double> &errors, const double thresh)
{
    Indices indices;
    int P = errors.size();
    for (unsigned int i = 0; i < P; i++)
        if (errors[i] < thresh)
            indices.push_back(i);
    return indices;
}

template <typename Datum, typename Param>
Indices ransac(
    const std::vector<Datum> &data,
    AbstractModel<Datum, Param> *model,
    const int n_inital_samples,
    const int iterations = 20,
    const int min_pts_as_good_fit = 10,
    const int dist_thresh_for_inlier_point = 2.0,
    const bool is_print = true)
{
    typedef std::vector<Datum> Data;

    Indices best_indices;
    unsigned int best_num_inlier_points = 0;
    RandomDataSampler<Datum> sampler(data);

    // Loop of {sample points; fit}
    for (int i = 0; i < iterations; i++)
    {
        // Sample data.
        const Data sample_data = sampler.sample(n_inital_samples);

        // Fit model.
        model->fit(sample_data);

        // Count inliers.
        const std::vector<double> errors = model->calcErrors(data);
        int num_pts_as_good_fit = countDataSmallerThan(
            errors, dist_thresh_for_inlier_point);

        // Print result.
        if (is_print)
        {
            std::cout << "Iteration " << i << ": " << std::endl;
            std::cout << "    num inliers: " << num_pts_as_good_fit << std::endl;
            std::cout << "    ";
            model->printParam();
        }

        // Check if there are enough inliers.
        if (num_pts_as_good_fit >= min_pts_as_good_fit)
        {
            // A good result has been found.

            /**
             * It would be better to fit the model again using inlier points.
             * TODO: Add this step.
             */

            if (num_pts_as_good_fit >= best_num_inlier_points)
            { // Update best indices.
                best_num_inlier_points = num_pts_as_good_fit;
                best_indices = getIndicesOfDataSmallerThan(
                    errors, dist_thresh_for_inlier_point);

                if (is_print)
                {
                    std::cout << "    This is a better model!!! "<< std::endl;
                }
            }
        }
    }
    if (best_indices.empty())
        return {};
    else
    {
        // Fit the model again using the inlier data.
        Data sample_data;
        for (int idx : best_indices)
            sample_data.push_back(data[idx]);
        model->fit(sample_data);

        // Return indices.
        return best_indices;
    }
}
} // namespace ransac
#endif
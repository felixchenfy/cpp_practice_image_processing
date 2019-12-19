#ifndef MY_RANSAC_RANSAC_H
#define MY_RANSAC_RANSAC_H

#include "my_ransac/abstract_model.hpp"

#include <iostream>
#include <vector>
#include <stdexcept>

template <class BidiIter>
BidiIter random_unique(BidiIter begin, BidiIter end, size_t num_random)
{
    /**
     * Randomly select num_random elements from [begin, end],
     *      and place them in [begin, begin + num_random).
     * Copied from here: https://stackoverflow.com/questions/9345087/choose-m-elements-randomly-from-a-vector-containing-n-elements
     */
    size_t left = std::distance(begin, end);
    while (num_random--)
    {
        BidiIter r = begin;
        std::advance(r, rand() % left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

template <typename Datum, typename Param>
std::vector<int> ransac(
    const std::vector<Datum> &data,
    models::AbstractModel<Datum, Param> *model,
    const int n_pts_to_fit_model,
    const int iterations = 20,
    const int min_pts_as_good_fit = 10,
    const int dist_thresh_for_inlier_point = 2.0, 

{
    std::vector<int> best_indices;
    Param best_param;
    for (int i = 0; i < iterations; i++)
    {
    }
    return best_indices;
}
#endif
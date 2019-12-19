#ifndef MY_RANSAC_RANDOM_DATA_SAMPLER_H
#define MY_RANSAC_RANDOM_DATA_SAMPLER_H

#include <iostream>
#include <vector>

template <typename Datum>
class RandomDataSampler
{
    typedef std::vector<Datum> Data;
    typedef std::vector<unsigned int> Indices;

private:
    const Data &data_;
    Indices indices_;

public:
    RandomDataSampler(const Data &data) : data_(data)
    {
        for (int i = 0; i < data.size(); i++)
            indices_.push_back(i);
    }
    Data sample(const int N)
    {
        this->random_unique<Indices::iterator>(
            indices_.begin(), indices_.end(), N);
        Data sampled_data;
        sampled_data.reserve(N);
        for (int i = 0; i < N; i++)
            sampled_data.push_back(data_[indices_[i]]);
        return sampled_data;
    }

private:
    template <class BidiIter>
    BidiIter random_unique(BidiIter begin, BidiIter end, size_t num_random) const
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
};

#endif

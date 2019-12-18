#ifndef MY_RANSAC_MODEL_H
#define MY_RANSAC_MODEL_H

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>
#include <stdexcept>

namespace models
{

template <typename Datum>
class AbstractModel
{
    typedef std::vector<Datum> Data;
    typedef std::vector<double> Param; // Parameter of the model
    typedef double Error;
    typedef std::vector<Error> Errors;

public:
    AbstractModel(const Data &  = {})
    {
        if (!data.empty())
            this->fitParam(data);
    }
    virtual ~AbstractModel() {}

public:
    virtual void fitParam(const Data &data) = 0;
    virtual Error calcError(const Datum &datum) = 0;
    Errors calcErrors(const Data &data)
    {
        Errors errors;
        errors.reserve(data.size());
        for (const Datum &data : data)
            errors.push_back(this->calcError(data));
        return errors;
    }
    Param param()
    {
        checkParam();
        return param_;
    };

protected:
    Param param_;
    void checkParam()
    {
        if (param_.empty())
            throw std::runtime_error("Model hasn't been fitted.");
    }
};

class ModelLine2D : public AbstractModel<cv::Point2i>
{
    typedef cv::Point2i Datum;
    typedef std::vector<Datum> Data;
    typedef std::vector<double> Param; // Parameter of the model
    typedef double Error;

public:
    ModelLine2D(const Data &data = {}) : AbstractModel(data) {}

    void fitParam(const Data &data)
    {
        if (data.size() != 2)
            throw std::runtime_error("W.");
    }
    Error calcError(const Datum &data)
    {
        this->checkParam();
    }
};

} // namespace models

#endif
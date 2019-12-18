#ifndef MY_RANSAC_MODEL_H
#define MY_RANSAC_MODEL_H

#include <iostream>
#include <vector>
#include <stdexcept>

namespace models
{

typedef std::vector<double> Data;  // shape=(N, ). N features. For a 2d point of (x, y), N=2.
typedef std::vector<Data> Datas;   // shape=(P, N). P points with N features.
typedef std::vector<double> Param; // Parameter of the model. Usually N+1 dimensions.
typedef double Error;
typedef std::vector<Error> Errors;

class AbstractModel
{

public:
    AbstractModel(const Datas &data = {})
    {
        if (!data.empty())
            this->fitParam(data);
    }
    virtual ~AbstractModel() {}

public:
    virtual void fitParam(const Datas &datas) = 0;
    virtual Error calcError(const Data &data) = 0;
    Errors calcErrors(const Datas &datas)
    {
        Errors errors;
        errors.reserve(datas.size());
        for (const Data &data : datas)
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

class ModelLine2D : public AbstractModel
{
private:
    // We use points to fit a 2D line. Each point has 2 features, x and y.
    const int FEATURE_DIM_ = 2;

    // We need at least 2 points to fit a line.
    const int MIN_DATA_FOR_FITTING_ = 2;

public:
    ModelLine2D(const Datas &data = {}) : AbstractModel(data) {}

    void fitParam(const Datas &datas)
    {
        if (datas.size() < MIN_DATA_FOR_FITTING_ || datas[0].size != FEATURE_DIM_)
            throw std::runtime_error("Invalid datas size.");
    }
    Error calcError(const Data &data)
    {
        this->checkParam();
    }
};

} // namespace models

#endif
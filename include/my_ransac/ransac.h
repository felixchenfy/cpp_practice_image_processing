#ifndef MY_RANSAC_RANSAC_H
#define MY_RANSAC_RANSAC_H

#include <iostream>
#include <vector>
#include <stdexcept>

typedef std::vector<double> Data;  // shape=(N, ). N features.
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
    virtual void fitParam(const Datas &datas);
    virtual Error calcError(const Data &data);
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

#endif
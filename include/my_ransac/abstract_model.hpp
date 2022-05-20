#ifndef MY_RANSAC_ABSTRACT_MODEL_H
#define MY_RANSAC_ABSTRACT_MODEL_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

template <typename Datum, typename Param>
class AbstractModel {
    typedef std::vector<Datum> Data;

   public:
    AbstractModel() {}
    virtual ~AbstractModel() {}

   public:
    virtual void fit(const Data &data) = 0;
    virtual Param getParam() const = 0;
    virtual void printParam() const = 0;
    virtual double calcError(const Datum &datum) const = 0;
    std::vector<double> calcErrors(const Data &data) const {
        std::vector<double> errors;
        errors.reserve(data.size());
        for (const Datum &data : data)
            errors.push_back(this->calcError(data));
        return errors;
    }

   protected:
    bool isFitted_ = false;
};

#endif
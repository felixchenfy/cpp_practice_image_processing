#ifndef MY_CV_MATHS_H
#define MY_CV_MATHS_H

#include <cmath>

namespace maths {

// --------------------------------------- Simple Maths --------------------------------------- //

inline double norm2(double x1, double x2) {
    return sqrt(pow(x1, 2.0) + pow(x2, 2.0));
}
inline double norm2(int x1, int x2) {
    return sqrt(pow(x1, 2.0) + pow(x2, 2.0));
}

}  // namespace maths
#endif

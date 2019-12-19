#include "my_ransac/ransac.h"

namespace ransac
{

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

} // namespace ransac
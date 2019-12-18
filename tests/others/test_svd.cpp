
/**
 * Reference: The code is copied from:
 * https://eigen.tuXdamily.org/dox/classEigen_1_1JacobiSVD.html
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
// #include <Eigen/Dense> // For initializing MatrixXd
#include <iostream>

using namespace std;

int main(int argc, char const *argv[])
{
  Eigen::MatrixXd m = Eigen::MatrixXd::Random(3, 2);
  // Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  cout << "Here is the matrix m:" << endl
       << m << endl;

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);

  cout << "Its singular values are:" << endl
       << svd.singularValues() << endl;
  cout << "Its left singular vectors are the columns of the thin U matrix:" << endl
       << svd.matrixU() << endl;
  cout << "Its right singular vectors are the columns of the thin V matrix:" << endl
       << svd.matrixV() << endl;

  Eigen::Vector3d rhs(1, 0, 0);
  cout << "Now consider this rhs vector:" << endl
       << rhs << endl;
  cout << "A least-squares solution of m*x = rhs is:" << endl
       << svd.solve(rhs) << endl;
  return 0;
}

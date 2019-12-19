
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
     // Eigen::MatrixXd mat = Eigen::MatrixXd::Random(3, 2);
     // Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(3, 2);
     // Eigen::Matrix3d mat = Eigen::Matrix3d::Zero();
     Eigen::MatrixXd mat(2, 2);
     mat(0, 0) = 1.0;
     mat(1, 0) = 1.0;
     mat(0, 1) = 5.0;
     mat(1, 1) = 5.0;

     cout << "Here is the matrix mat:" << endl
          << mat << endl;

     Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
     Eigen::Vector2d s = svd.singularValues();
     Eigen::Matrix<double, 2, 2> u = svd.matrixU();
     Eigen::Matrix<double, 2, 2> v = svd.matrixV();
     cout << "Its singular values are:" << endl
          << svd.singularValues() << endl;
     cout << "Its left singular vectors are the columns of the thin U matrix:" << endl
          << svd.matrixU() << endl;
     cout << "Its right singular vectors are the columns of the thin V matrix:" << endl
          << svd.matrixV() << endl;

     // Eigen::Vector3d rhs(1, 0, 0);
     Eigen::Vector2d rhs(1, 0);
     cout << "Now consider this rhs vector:" << endl
          << rhs << endl;
     cout << "A least-squares solution of mat*x = rhs is:" << endl
          << svd.solve(rhs) << endl;

     Eigen::Vector2d mean = mat.colwise().mean();     // 1, 5
     Eigen::Vector2d mean_row = mat.rowwise().mean(); // 3, 3
     typedef Eigen::Matrix<double, 1, 2> Mat12d;
     Mat12d mean_col = mat.colwise().mean(); // 1, 5

     cout << "Column wise mean: " << mean << endl;
     cout << "Row wise mean: " << mean_row << endl;
     cout << "Subtraction by colwise(): " << mat.colwise() - mean << endl;
     cout << "Subtraction by rowwise(): " << mat.rowwise() - mean.transpose() << endl;

     Eigen::Vector2d first_column = mat.col(0);
     cout << "\n first_column: " << first_column << endl;
     double norm_of_first_column = first_column.dot(first_column);
     cout << "\n norm_of_first_column: " << norm_of_first_column << endl;
     
     return 0;
}

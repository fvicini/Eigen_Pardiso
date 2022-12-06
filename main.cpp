
#include "Eigen/Eigen"

#include <iostream>
#include <Macro.hpp>

#if ENABLE_MKL
#include <Eigen/PardisoSupport>
#endif

// ***************************************************************************
int main(int argc, char** argv)
{
  Eigen::Vector3d test(1.0, 2.0, 3.0);

  std::cerr<< "Test "<< test.transpose()<< std::endl;

  Eigen::SparseMatrix<double> A;
  A.resize(2, 2);

  std::list<Eigen::Triplet<double>> triplets;

  triplets.push_back(Eigen::Triplet<double>(0, 0, 17));
  triplets.push_back(Eigen::Triplet<double>(0, 1, 38));
  triplets.push_back(Eigen::Triplet<double>(1, 0, 38));
  triplets.push_back(Eigen::Triplet<double>(1, 1, 85));


  A.setFromTriplets(triplets.begin(),
                    triplets.end());

  Eigen::VectorXd b;
  b.resize(2);
  b[0] = 55;
  b[1] = 123;

  Eigen::VectorXd x;

#if ENABLE_MKL
  PardisoLU< SparseMatrix<double> > solver( A );
  x = solver.solve(b);
#else
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(A);
  x = solver.solve(b);
#endif

  std::cerr<< "The solution is: "<< x.transpose()<< std::endl;

  return 0;
}
// ***************************************************************************

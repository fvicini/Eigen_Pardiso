
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

  std::cerr.precision(16);
  std::cerr<< std::scientific<< "Test "<< test.transpose()<< std::endl;

  Eigen::SparseMatrix<double> A;
  A.resize(2, 2);

  std::list<Eigen::Triplet<double>> triplets;

  triplets.push_back(Eigen::Triplet<double>(0, 0, 17.0));
  triplets.push_back(Eigen::Triplet<double>(0, 1, 38.0));
  triplets.push_back(Eigen::Triplet<double>(1, 0, 38.0));
  triplets.push_back(Eigen::Triplet<double>(1, 1, 85.0));


  A.setFromTriplets(triplets.begin(),
                    triplets.end());
  A.makeCompressed();

  Eigen::VectorXd b;
  b.resize(2);
  b[0] = 55.0;
  b[1] = 123.0;

  Eigen::VectorXd x;

#if ENABLE_MKL
  Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver(A);
  const Eigen::ComputationInfo solverResult = solver.info();
  if (solverResult != Eigen::ComputationInfo::Success)
  {
    throw std::runtime_error("Error " +
                             std::to_string(static_cast<unsigned int>(solverResult)) +
                             " in eigen solver");
  }

  x = solver.solve(b);
#else
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(A);
  const Eigen::ComputationInfo solverResult = solver.info();
  if (solverResult != Eigen::ComputationInfo::Success)
  {
    throw std::runtime_error("Error " +
                             std::to_string(static_cast<unsigned int>(solverResult)) +
                             " in eigen solver");
  }

  x = solver.solve(b);
#endif

  std::cerr<< std::scientific<< "The solution is: "<< x.transpose()<< std::endl;

  return 0;
}
// ***************************************************************************


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

  Eigen::SparseMatrix<double, 0, MKL_INT64> A;
  A.resize(2, 2);

  std::list<Eigen::Triplet<double>> triplets;

  triplets.push_back(Eigen::Triplet<double>(0, 0, 17.0));
  triplets.push_back(Eigen::Triplet<double>(0, 1, 38.0));
  triplets.push_back(Eigen::Triplet<double>(1, 0, 38.0));
  triplets.push_back(Eigen::Triplet<double>(1, 1, 85.0));

  A.setFromTriplets(triplets.begin(),
                    triplets.end());
  A.makeCompressed();

  Eigen::VectorXd b = Eigen::VectorXd::Zero(2);
  b[0] = A.row(0).sum();
  b[1] = A.row(1).sum();

  Eigen::VectorXd x_Pardiso = Eigen::VectorXd::Zero(2);

#if ENABLE_MKL
  //Eigen::PardisoLU<Eigen::SparseMatrix<double, 0, MKL_INT64>> solver_Pardiso;
  Eigen::PardisoLDLT<Eigen::SparseMatrix<double, 0, MKL_INT64>> solver_Pardiso(A);

  solver_Pardiso.compute(A);
  const Eigen::ComputationInfo solverResult_Pardiso = solver_Pardiso.info();
  if (solverResult_Pardiso != Eigen::ComputationInfo::Success)
  {
    throw std::runtime_error("Error " +
                             std::to_string(static_cast<unsigned int>(solverResult_Pardiso)) +
                             " in eigen Pardiso solver");
  }

  x_Pardiso = solver_Pardiso.solve(b);
#endif

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, 0, MKL_INT64>> solver_Eigen(A);
  const Eigen::ComputationInfo solverResult_Eigen = solver_Eigen.info();
  if (solverResult_Eigen != Eigen::ComputationInfo::Success)
  {
    throw std::runtime_error("Error " +
                             std::to_string(static_cast<unsigned int>(solverResult_Eigen)) +
                             " in eigen Eigen solver");
  }

  Eigen::VectorXd x_Eigen = solver_Eigen.solve(b);

  const Eigen::VectorXd x_Solution = Eigen::VectorXd::Ones(2);

  std::cerr<< std::scientific<< "The Pardiso solution is: "<< x_Pardiso.transpose()<< std::endl;
  std::cerr<< std::scientific<< "The Pardiso error is: "<< (x_Pardiso - x_Solution).norm()<< std::endl;

  std::cerr<< std::scientific<< "The Eigen solution is: "<< x_Eigen.transpose()<< std::endl;
  std::cerr<< std::scientific<< "The Eigen error is: "<< (x_Eigen - x_Solution).norm()<< std::endl;

  return 0;
}
// ***************************************************************************

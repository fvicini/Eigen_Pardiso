
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

  return 0;
}
// ***************************************************************************

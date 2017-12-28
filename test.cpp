#include "activations.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include <string>

/* TODO: use expected arrays for the results */
bool test_relu() {
  /* Test ReLU forwards. */
  ReLU relu;
  xt::xarray<double> inp {1.0, -2.0, 3.0, 0.0};
  xt::xarray<double> fwd_result = relu.forward(inp);

  std::cout << "FORWARDS (input, result)" << std::endl;
  std::cout << inp << std::endl;
  std::cout << fwd_result << std::endl;

  /* Test ReLU backwards. */
  xt::xarray<double> inc_grad {3.0, 0.5, -2.0, 1.0};
  xt::xarray<double> bwd_result = relu.backward(inc_grad);

  std::cout << "BACKWARDS (inc grad, result)" << std::endl;
  std::cout << inc_grad << std::endl;
  std::cout << bwd_result << std::endl;

  return true;
}

/* TODO: use expected arrays for the results */
bool test_sigmoid() {
  /* Test Sigmoid forwards. */
  Sigmoid sig;
  xt::xarray<double> inp {1.0, -2.0, 3.0, 0.0};
  xt::xarray<double> fwd_result = sig.forward(inp);

  std::cout << "FORWARDS (input, result)" << std::endl;
  std::cout << inp << std::endl;
  std::cout << fwd_result << std::endl;

  /* Test Sigmoid backwards. */
  xt::xarray<double> inc_grad {3.0, 0.5, -2.0, 1.0};
  xt::xarray<double> bwd_result = sig.backward(inc_grad);

  std::cout << "BACKWARDS (inc grad, result)" << std::endl;
  std::cout << inc_grad << std::endl;
  std::cout << bwd_result << std::endl;

  return true;
}

/* Runs a given test and reports the results. */
void run_test(bool (*test)(void), std::string test_name) {
  if (test()) {
    std::cout << test_name << " Tests PASSED" << std::endl;
  } else {
    std::cout << test_name << " Tests FAILED" << std::endl;
  }
}

int main() {
  run_test(test_relu, "ReLU"); 
  run_test(test_sigmoid, "Sigmoid"); 
  return 0;
}

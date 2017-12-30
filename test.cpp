#include "activations.h"
#include "graph.h"
#include "layers.h"
#include "losses.h"
#include "optimizers.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include <string>

void graph_test_on_input(Graph &graph, xt::xarray<double> test) {
  std::cout << "Testing On: " << test << std::endl;
  std::cout << "Result: " << graph.run(test) << std::endl;
} 

bool test_graph_optimization() {
  Graph graph;
  SGDOptimizer sgd(0.05);
  graph.add_layer(new FullyConnected(3, 3, &sgd)); 

  MSELoss mse(3);
  xt::xarray<double> inp {{1.0, -2.0, 3.0}};
  xt::xarray<double> target {{2.0, -4.0, 6.0}};

  graph.optimize(&mse, inp, target, 20);

  xt::xarray<double> test1 {{1.0, -2.0, 3.0}};
  graph_test_on_input(graph, test1);
 
  xt::xarray<double> test2 {{3.0, 24.0, -5.0}};
  graph_test_on_input(graph, test2);

  return true;
}

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
bool test_mse_loss() {
  /* Test MSE loss. */
  MSELoss mse(3);
  
  xt::xarray<double> inp {{1.0, -3.0, 3.0}};
  xt::xarray<double> targets {{2.0, -1.0, 3.0}};
  xt::xarray<double> fwd_result = mse.forward(inp, targets);

  std::cout << "FORWARDS (input, targets, result)" << std::endl;
  std::cout << inp << std::endl;
  std::cout << targets << std::endl;
  std::cout << fwd_result << std::endl;

  /* Test FC backwards. */ 
  xt::xarray<double> bwd_result = mse.backward();

  std::cout << "BACKWARDS (result)" << std::endl;
  std::cout << bwd_result << std::endl;

  return true;
}

/* TODO: use expected arrays for the results */
bool test_fc() {
  /* Test FC forwards. */
  SGDOptimizer sgd(0.03);
  FullyConnected fc(3, 5, &sgd); 
  
  xt::xarray<double> inp {{1.0, -2.0, 3.0}};
  xt::xarray<double> fwd_result = fc.forward(inp);

  std::cout << "FORWARDS (input, result)" << std::endl;
  std::cout << inp << std::endl;
  std::cout << fwd_result << std::endl;

  /* Test FC backwards. */ 
  xt::xarray<double> inc_grad {{3.0, 0.5, -2.0, 1.0, 0.0}};
  xt::xarray<double> bwd_result = fc.backward(inc_grad);

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
  run_test(test_fc, "Fully Connected"); 
  run_test(test_mse_loss, "MSE Loss"); 
  run_test(test_graph_optimization, "Graph Optimization"); 
  return 0;
}

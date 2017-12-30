#include "graph.h"
#include "xtensor/xio.hpp"

void Graph::add_layer(Layer *layer) {
  this->layers.push_back(layer);
}

xt::xarray<double> Graph::run(xt::xarray<double> input) {
  xt::xarray<double> result;

  for (size_t i = 0; i < this->layers.size(); i++) {
    result = this->layers[i]->forward(input);
    input = result;
  }

  return result;
}

void Graph::backwards(xt::xarray<double> loss_grad) {
  xt::xarray<double> grad = loss_grad;

  for (size_t i = 0; i < this->layers.size(); i++) {
    grad = this->layers[this->layers.size() - i - 1]->backward(grad);
  }
}

void Graph::optimize(Loss *loss, xt::xarray<double> input, xt::xarray<double> target, size_t num_iter) {

  for (size_t i = 0; i < num_iter; i++) {
    xt::xarray<double> result = this->run(input);
    xt::xarray<double> loss_amt = loss->forward(result, target);
    std::cout << "Loss: " << loss_amt << std::endl;
    xt::xarray<double> loss_grad = loss->backward();
    this->backwards(loss_grad);
  }
}

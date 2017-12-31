#include "layers.h"
#include "xtensor/xrandom.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor-blas/xlinalg.hpp"

FullyConnected::FullyConnected(int num_input, int num_output, Optimizer *optimizer) {
  this->weights = xt::random::randn<double>({num_input, num_output});
  this->biases = xt::random::randn<double>({num_output});
  this->optimizer = optimizer;
}

xt::xarray<double> FullyConnected::forward(xt::xarray<double> input) {
  this->input = input;
  return xt::linalg::dot(input, this->weights) + this->biases;
}

xt::xarray<double> FullyConnected::backward(xt::xarray<double> incoming_grad) {
  /* y = xW + b */
  /* dW = x'dy */
  /* dx = dy * W'*/

  xt::xarray<double> dW = xt::linalg::dot(xt::transpose(this->input), incoming_grad); 
  xt::xarray<double> db = xt::sum(incoming_grad, {0});

  this->optimizer->update(this->weights, dW);
  this->optimizer->update(this->biases, db);

  return xt::linalg::dot(incoming_grad, xt::transpose(this->weights));
}

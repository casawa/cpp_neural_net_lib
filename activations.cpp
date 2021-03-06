#include "activations.h"
#include "xtensor/xmath.hpp"

/* TODO test sigmoid more */
xt::xarray<double> Sigmoid::forward(xt::xarray<double> input) {
  this->output = 1 / (1 + xt::exp(-1.0 * input));
  return this->output;;
}

xt::xarray<double> Sigmoid::backward(xt::xarray<double> incoming_grad)  {
  return incoming_grad * this->output * (1 - this->output);
}

/* TODO test ReLU more */
xt::xarray<double> ReLU::forward(xt::xarray<double> input) {
  this->output = xt::maximum(input, 0);
  return this->output;
}

xt::xarray<double> ReLU::backward(xt::xarray<double> incoming_grad) {
  return xt::sign(this->output) * incoming_grad;
}

/* TODO test Tanh more */
xt::xarray<double> Tanh::forward(xt::xarray<double> input) {
  this->output = xt::tanh(input);
  return this->output;
}

xt::xarray<double> Tanh::backward(xt::xarray<double> incoming_grad) {
  return (1 - xt::pow(this->output, 2)) * incoming_grad;
}

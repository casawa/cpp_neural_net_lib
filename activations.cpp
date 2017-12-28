#include "activations.h"

/* TODO */
xt::xarray<double> Sigmoid::forward(xt::xarray<double> input) {
  return NULL;
}

/* TODO */
xt::xarray<double> Sigmoid::backward(xt::xarray<double> incoming_grad)  {
  return NULL;
}

/* TODO test ReLU more */
xt::xarray<double> ReLU::forward(xt::xarray<double> input) {
  this->output = xt::maximum(input, 0);
  return this->output;
}

xt::xarray<double> ReLU::backward(xt::xarray<double> incoming_grad) {
  return xt::sign(this->output) * incoming_grad;
}


void sigmoid() {}
void relu() {}

#include "losses.h"

MSELoss::MSELoss(int num_targets) {
  this->num_targets = num_targets;
}

xt::xarray<double> MSELoss::forward(xt::xarray<double> input, xt::xarray<double> target) {
  this->input = input;
  this->target = target;
  return xt::mean(xt::pow(input - target, 2)); 
}

xt::xarray<double> MSELoss::backward() {
  return (2.0 / this->num_targets) * (input - target); 
}

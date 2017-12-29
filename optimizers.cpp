#include "optimizers.h"

/* TODO use initializer list? */
Optimizer::Optimizer(double learning_rate) {
  this->learning_rate = learning_rate;
}

void SGDOptimizer::update(xt::xarray<double> &weights, const xt::xarray<double> &grad) {
  weights -= this->learning_rate * grad;
}

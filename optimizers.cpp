#include "optimizers.h"
#include <math.h>
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"

/* TODO use initializer list? */
Optimizer::Optimizer(double learning_rate) {
  this->learning_rate = learning_rate;
}

void SGDOptimizer::update(xt::xarray<double> &weights, const xt::xarray<double> &grad) {
  weights -= this->learning_rate * grad;
}

MomentumOptimizer::MomentumOptimizer(double learning_rate, double momentum) : Optimizer(learning_rate) {
  this->momentum = momentum;
}

void MomentumOptimizer::update(xt::xarray<double> &weights, const xt::xarray<double> &grad) {
  this->velocity = this->momentum * this->velocity - this->learning_rate * grad;
  weights += this->velocity;
}

AdamOptimizer::AdamOptimizer(double learning_rate, double beta1, double beta2, double eps) : Optimizer(learning_rate) {
  this->beta1 = beta1;
  this->beta2 = beta2;
  this->eps = eps;
}

void AdamOptimizer::update(xt::xarray<double> &weights, const xt::xarray<double> &grad) {
  this->t += 1;
  this->first_moment = this->beta1 * this->first_moment + (1 - this->beta1) * grad;
  this->second_moment = this->beta2 * this->second_moment + (1 - this->beta2) * grad * grad;
  xt::xarray<double> bias_corrected_first_moment = this->first_moment / (1 - pow(this->beta1, t)); 
  xt::xarray<double> bias_corrected_second_moment = this->second_moment / (1 - pow(this->beta2, t)); 

  weights -= this->learning_rate * bias_corrected_first_moment / (xt::sqrt(bias_corrected_second_moment) + this->eps);
}

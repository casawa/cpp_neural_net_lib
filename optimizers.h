/* Optimizers. */
#ifndef OPTIMIZERS_H 
#define OPTIMIZERS_H 

#include "xtensor/xarray.hpp"

/* Optimization Base Class. */
class Optimizer {
  protected:
    double learning_rate;
  public:
    Optimizer(double learning_rate);
    virtual void update(xt::xarray<double> &weights, const xt::xarray<double> &grad) = 0;
};

class SGDOptimizer: public Optimizer {
  public:
    using Optimizer::Optimizer;
    void update(xt::xarray<double> &weights, const xt::xarray<double> &grad);
};

#endif

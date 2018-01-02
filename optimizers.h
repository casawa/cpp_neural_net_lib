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

class MomentumOptimizer: public Optimizer {
  private:
    double momentum;
    xt::xarray<double> velocity;
  public:
    MomentumOptimizer(double learning_rate, double momentum=0.9);
    void update(xt::xarray<double> &weights, const xt::xarray<double> &grad);
};

/* Based on Kingma et al., 2014 */
class AdamOptimizer: public Optimizer {
  private:
    xt::xarray<double> first_moment;
    xt::xarray<double> second_moment;
    int t = 0;

    double beta1, beta2;
    double eps;
  public:
    AdamOptimizer(double learning_rate, double beta1=0.9, double beta2=0.999, double eps=1e-8);
    void update(xt::xarray<double> &weights, const xt::xarray<double> &grad);
};

#endif

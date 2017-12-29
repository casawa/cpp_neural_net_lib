/* Optimizers. */
#ifndef OPTIMIZERS_H 
#define OPTIMIZERS_H 

/* Optimization Base Class. */
class Optimizer {
  private:
    double learning_rate;
  public:
    Optimizer(double learning_rate);
    virtual void update(xt::xarray<double> &weights, const xt::xarray<double> &grad) = 0;
};

class SGD: public Optimizer {
  public:
    void update(xt::xarray<double> &weights, const xt::xarray<double> &grad);
};

#endif

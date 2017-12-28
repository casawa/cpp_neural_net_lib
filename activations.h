/* Non-Linear Activations. */
#ifndef ACTIVATIONS_H 
#define ACTIVATIONS_H 

#include "src/layer.h"

class Sigmoid: public Activation {
  public:
    xt::xarray<double> forward(xt::xarray<double> input);
    xt::xarray<double> backward(xt::xarray<double> incoming_grad);
};

class ReLU: public Activation {
  private:
    xt::xarray<double> output;
  public:
    xt::xarray<double> forward(xt::xarray<double> input);
    xt::xarray<double> backward(xt::xarray<double> incoming_grad);
};

void sigmoid();
void relu();

#endif

#ifndef LAYER_H 
#define LAYER_H 
#include "xtensor/xarray.hpp"

class Layer {
  public:
    virtual xt::xarray<double> forward(xt::xarray<double> input) = 0;
    virtual xt::xarray<double> backward(xt::xarray<double> incoming_grad) = 0;
};

class Activation: public Layer {};

class Loss {
  public:
    virtual xt::xarray<double> forward(xt::xarray<double> input, xt::xarray<double> target) = 0;
    virtual xt::xarray<double> backward() = 0;
};

#endif

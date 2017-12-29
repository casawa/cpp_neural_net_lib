/* Linear Layers. */
#ifndef LAYERS_H 
#define LAYERS_H 

#include "src/layer.h"
#include "optimizers.h"

class FullyConnected: public Layer {
  private:
    xt::xarray<double> input;
    xt::xarray<double> weights;
    xt::xarray<double> biases;
    Optimizer *optimizer;
  public:
    FullyConnected(int num_input, int num_output, Optimizer *optimizer);
    xt::xarray<double> forward(xt::xarray<double> input);
    xt::xarray<double> backward(xt::xarray<double> incoming_grad);
};

#endif

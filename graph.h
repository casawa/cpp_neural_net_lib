/* The Model Graph */
#ifndef GRAPH_H 
#define GRAPH_H 

#include "src/layer.h"
#include "xtensor/xarray.hpp"
#include <vector>

class Graph {
  private:
    std::vector<Layer *> layers;
    void backwards(xt::xarray<double> loss_grad);
  public:
    void add_layer(Layer *layer);
    void optimize(Loss *loss, xt::xarray<double> input, xt::xarray<double> target, size_t num_iter);
    xt::xarray<double> run(xt::xarray<double> input);
};

#endif

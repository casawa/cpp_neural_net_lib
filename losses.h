/* Losses. */
#ifndef LOSSES_H 
#define LOSSES_H 

#include "src/layer.h"

/* Mean Squared Error Loss. */
class MSELoss: public Loss {
  private:
    int num_targets;
    xt::xarray<double> input;
    xt::xarray<double> target;
  public:
    MSELoss(int num_targets);
    xt::xarray<double> forward(xt::xarray<double> input, xt::xarray<double> target);
    xt::xarray<double> backward();
};

#endif

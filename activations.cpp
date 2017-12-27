#include "activations.h"
#include "activation.h"

class Sigmoid: public Activation {
  public:
    void forward();
    void backward();
};

class ReLU: public Activation {
  public:
    void forward();
    void backward();
};

void Sigmoid::forward() {

}

void Sigmoid::backward() {

}

void ReLU::forward() {

}

void ReLU::backward() {

}


void sigmoid() {}
void relu() {}

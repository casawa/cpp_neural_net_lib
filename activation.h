#ifndef ACTIVATION_H 
#define ACTIVATION_H 

class Activation {
  public:
    virtual void forward() = 0;
    virtual void backward() = 0;
};

#endif

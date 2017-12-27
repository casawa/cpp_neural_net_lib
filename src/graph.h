#include "layer.h"
#include <vector>

class Graph {
  private:
    std::vector<Layer *> layers;
    void full_forward();
    void full_backward();
  public:
    void add_to_graph(Layer *layer);
};

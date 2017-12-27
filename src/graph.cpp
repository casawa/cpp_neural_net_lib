#include "graph.h"

void Graph::add_to_graph(Layer *layer) {
  this->layers.push_back(layer);
}

void Graph::full_forward() {
  for (size_t i = 0; i < layers.size(); i++) {
    /* TODO take in input and pass to next layer */
    layers[i]->forward();
  }
}

void Graph::full_backward() {
  for (size_t i = 0; i < layers.size(); i++) {
    /* TODO take in incoming gradient */
    layers[layers.size() - i - 1]->backward();
  }
}

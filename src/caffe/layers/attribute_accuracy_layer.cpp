#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AttributeAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //top_k_ = this->layer_param_.accuracy_param().top_k();
}

template <typename Dtype>
void AttributeAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int dim = bottom[0]->count() / bottom[0]->num();
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), dim);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  top[0]->Reshape(1, 1, 1, dim);
}

template <typename Dtype>
void AttributeAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / bottom[0]->num();
  vector<Dtype> attr_acc(dim,0);
  vector<Dtype> num_valid(dim,0);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
        if (bottom_label[i*dim+j]>=0) { // ignore negative labels
//            printf("Label is: %f\n", bottom_label[i*dim+j]);
//            printf("Prediction is: %f\n", bottom_data[i*dim+j]);
            ++num_valid[j];
            attr_acc[j] += (bottom_data[i*dim+j] == bottom_label[i*dim+j]);
        }
    }
  }
  //printf("Accuracy: ");i
  for (int j=0; j<dim; ++j) {
    top[0]->mutable_cpu_data()[j] = attr_acc[j] / std::max(Dtype(1),num_valid[j]);
    //printf("%f ", top[0]->mutable_cpu_data()[j]);
  }
  //printf("\n");
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AttributeAccuracyLayer);
REGISTER_LAYER_CLASS(AttributeAccuracy);
}  // namespace caffe


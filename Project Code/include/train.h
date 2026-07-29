#ifndef TRAIN_H
#define TRAIN_H

#include "nn.h"

/* Sequential training with tiled GEMM. Returns training time or -1 on error. */
double train_sequential(NeuralNetwork* nn,
                           float** inputs,
                           float** targets,
                           int num_samples,
                           int epochs,
                           int batch_size,
                           int report_loss_each_epoch);

/* Sequential training with naive GEMM. Returns training time or -1 on error. */
double train_sequential_naive(NeuralNetwork* nn,
                                 float** inputs,
                                 float** targets,
                                 int num_samples,
                                 int epochs,
                                 int batch_size,
                                 int report_loss_each_epoch);

/* OpenMP training with tiled GEMM. Returns training time or -1 on error. */
double train_openmp(NeuralNetwork* nn,
                       float** inputs,
                       float** targets,
                       int num_samples,
                       int epochs,
                       int batch_size,
                       int report_loss_each_epoch);

#endif

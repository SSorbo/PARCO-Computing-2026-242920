#ifndef TRAIN_MPI_H
#define TRAIN_MPI_H

#include <mpi.h>
#include "nn.h"

/* Synchronous data-parallel training. Returns local training time or -1. */
double train_mpi(NeuralNetwork *nn,
                    float **inputs,
                    float **targets,
                    int local_num_samples,
                    int epochs,
                    int batch_size,
                    int tile_size,
                    int report_loss_each_epoch,
                    MPI_Comm comm);

#endif

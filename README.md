# ESSD-ANET

The ANET model training, and calibrated ensemble generation scripts.

This code is provided as supplementary material with:

...: The EUPPBench postprocessing benchmark dataset v1.0, ...

**Please cite this article if you use (a part of) this code for a publication.**

## Required Python libraries

Pytorch (validated with version 1.13.0)
Numpy   (validated with version 1.23.3)
xarray  (validated version 2022.11.0)
netCDF4 (validated with version 1.5.7)

**Before conducting training and ensemble generation modify the train.py and generate.py scripts by setting the CUDA flag at the top of the scripts appropriately. If you wish to use a CUDA enabled device set CUDA = True.**

## Training the model

To initiate the training execute the train.py script in the following way:

python3 train.py ***\<path to data folder\>*** ***\<optional postfix for the output folder\>***

(You can acquire the data from https://github.com/EUPP-benchmark/ESSD-benchmark-datasets)

After training is concluded the folder named Model\_***\<training parameters, timestamp, optional postfix\>*** will contain two ANET models (best training loss and best validation loss models).

## Generating a calibrated ensemble

To generate the calibrated ensemble, execute:

python3 generate.py ***\<path to data folder\>*** ***\<path to model folder\>***

This will create a folder named Model\_***\<training parameters, timestamp, optional postfix\>***\_ensemble" containing .npy and .nc files each with the 51 member calibrated ensemble.

## Training time

Roughly 330 epochs, ~1 hour on a NVIDIA RTX 2080Ti graphics card.

Author: Peter Mlakar

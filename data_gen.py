import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='Data Generator')

parser.add_argument('-i',
                        '--input-data-file',
                        metavar='input_data_file',
                        type=str,
                        default='rbc_data.pt',
                        help='Input Data File'
                    )

parser.add_argument('-s',
                        '--samples',
                        metavar='samples',
                        type=int,
                        default=1510-100,
                        help='Number of samples to generate train/test/val datasets.'
                    )

args = parser.parse_args()

input_data_file=args.input_data_file
samples=args.samples

# read data
data = torch.load(input_data_file)

# standardization
std = torch.std(data)
avg = torch.mean(data)
data = (data - avg)/std
data = data[:,:,::4,::4]

# divide each rectangular snapshot into 7 subregions
# data_prep shape: num_subregions * time * channels * w * h
data_prep = torch.stack([data[:,:,:,k*64:(k+1)*64] for k in range(7)])

# use sliding windows to generate 9870 samples
# training 6000, validation 2000, test 1870
for j in range(0, samples):
    for i in range(7):
        torch.save(torch.FloatTensor(data_prep[i, j : j + 100]), "sample_" + str(j*7+i) + ".pt")

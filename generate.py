import torch
import numpy as np

from os.path import exists, join
from os import mkdir

import sys

CUDA = False

#########################################################

if len(sys.argv) < 2:

    print("Usage: python3 <path to data> <path to model>")
    exit()

from dataset import Databank, Dataset, norm, normalize, denormalize, load_training_dataset, load_validation_dataset

import netCDF4

#########################################################

PATH  = sys.argv[1]
MODEL_PATH = sys.argv[2]

#### Load training data, remove nan ####

_, _,\
_, _, _, _,\
_, _, _, _,\
X_mean, X_std,\
Y_mean, Y_std,\
P_alt_mean_md, P_alt_std_md,\
P_lat_mean_md, P_lat_std_md,\
P_lon_mean_md, P_lon_std_md,\
P_lnd_mean_md, P_lnd_std_md,\
P_alt_mean_st, P_alt_std_st,\
P_lat_mean_st, P_lat_std_st,\
P_lon_mean_st, P_lon_std_st,\
P_lnd_mean_st, P_lnd_std_st,\
_, _ = load_training_dataset(PATH)

#### Load validation data ####

VX, VY,\
VP_alt_md, VP_lat_md, VP_lon_md, VP_lnd_md,\
VP_alt_st, VP_lat_st, VP_lon_st, VP_lnd_st,\
time, stations = load_validation_dataset(PATH)

#### Load validation data ####

VX = normalize(VX, X_mean, X_std)
VY = normalize(VY, Y_mean, Y_std)

VP_alt_md = normalize(VP_alt_md, P_alt_mean_md, P_alt_std_md)
VP_lat_md = normalize(VP_lat_md, P_lat_mean_md, P_lat_std_md)
VP_lon_md = normalize(VP_lon_md, P_lon_mean_md, P_lon_std_md)
VP_lnd_md = normalize(VP_lnd_md, P_lnd_mean_md, P_lnd_std_md)

VP_alt_st = normalize(VP_alt_st, P_alt_mean_st, P_alt_std_st)
VP_lat_st = normalize(VP_lat_st, P_lat_mean_st, P_lat_std_st)
VP_lon_st = normalize(VP_lon_st, P_lon_mean_st, P_lon_std_st)
VP_lnd_st = normalize(VP_lnd_st, P_lnd_mean_st, P_lnd_std_st)

#########################################################
 
BATCH_SIZE = 512
MODEL_PATH = join(sys.argv[2], f"Model_valid")
EVALUATION_OUTPUT = f"{sys.argv[2]}_ensemble"

if not exists(EVALUATION_OUTPUT):
    mkdir(EVALUATION_OUTPUT)

#########################################################

bank = Databank(VX, VY, [VP_lat_md, VP_lon_md, VP_lnd_md, VP_alt_md, VP_lat_st, VP_lon_st, VP_lnd_st, VP_alt_st], cuda = CUDA)

D = Dataset(bank, bank.index, batch_size = BATCH_SIZE, train = False, cuda = CUDA)
M = torch.load(MODEL_PATH)
M = M.cuda() if CUDA else M.cpu()
M.eval()

P = np.zeros(VY.shape, dtype = np.float32)
S = np.zeros(VY.shape, dtype = np.float32)
ENS_C = np.zeros(VX.shape, dtype = np.float32)

with torch.no_grad():
    
    for i in range(len(D)):

        x, p, y, idx = D.__getitem__(i)

        o = torch.squeeze(M(x, p))

        P[idx[0, :], idx[1, :], :] = o[:, :, 0].cpu().detach().numpy()
        S[idx[0, :], idx[1, :], :] = o[:, :, 1].cpu().detach().numpy()
    
        print(f"{i + 1}/{len(D)}: {o.shape}")

print(P.shape)
print(S.shape)
print(VY.shape)
print(VX.shape)

P = denormalize(P, Y_mean, Y_std)
S = S*Y_std
VY = denormalize(VY, Y_mean, Y_std)
VX = denormalize(VX, X_mean, X_std)

for station in range(VX.shape[0]):
    for day in range(VX.shape[1]):
        for lead in range(VX.shape[2]):

            ens_sample = VX[station, day, lead]
            net_sample = np.sort(np.random.normal(P[station, day, lead], S[station, day, lead], size = VX.shape[-1]))

            idx = np.argsort(ens_sample)
            idx[np.argsort(ens_sample)] = np.arange(ens_sample.size)

            net_sample = net_sample[idx]

            ENS_C[station, day, lead, :] = net_sample

np.save(join(EVALUATION_OUTPUT, "pp_ensemble"), ENS_C)

# Write netCDF4 file #

TIER = 1
INSTITUTION = "ARSO"
EXPERIMENT = "ESSD-benchmark"
MODEL = "ANET"
VERSION = "v1.0"

netcdf = netCDF4.Dataset(join(EVALUATION_OUTPUT, f"{TIER}_{EXPERIMENT}_{INSTITUTION}_{MODEL}_{VERSION}.nc"), mode = "w", format = "NETCDF4_CLASSIC")

netcdf.createDimension("station_id", len(stations))
netcdf.createDimension("number", 51)
netcdf.createDimension("step", 21)
netcdf.createDimension("time", len(time))

t2m = netcdf.createVariable("t2m", np.float32, ("station_id", "time", "step", "number"))

t2m.institution = INSTITUTION
t2m.tier = TIER
t2m.experiment = EXPERIMENT
t2m.model = MODEL
t2m.version = VERSION
t2m.output = "members"

t2m[:, :, :, :] = ENS_C

netcdf.createVariable("model_altitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("model_latitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("model_longitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("model_land_usage", np.float32, ("station_id"), fill_value = np.nan)

netcdf.createVariable("station_altitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("station_latitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("station_longitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("station_land_usage", np.float32, ("station_id"), fill_value = np.nan)

print(netcdf)

netcdf.close()

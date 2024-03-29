import xarray as xr
import numpy as np
import torch

from os.path import join

CUDA = torch.cuda.is_available()

#### Dataset definition ####

class Databank():

    def __init__(self, X, Y, P, cuda = CUDA):

        self.X = torch.tensor(X, dtype = torch.float32)
        self.Y = torch.tensor(Y, dtype = torch.float32)
        self.P = torch.tensor(np.swapaxes(np.concatenate(P, axis = 0), 0, 1))
        self.n_samples = X.shape[0]*X.shape[1]
        self.n_members = X.shape[-1]

        self.index = np.stack(np.meshgrid(np.arange(0, X.shape[0]), np.arange(0, X.shape[1])), axis = 0)
        self.index = np.reshape(self.index, (self.index.shape[0], np.prod(self.index.shape[1:])))

        if cuda:

            self.X = self.X.cuda()
            self.Y = self.Y.cuda()
            self.P = self.P.cuda()

class Dataset(torch.utils.data.Dataset):

    def __init__(self, databank, index, n_predictors = 8, batch_size = 32, cuda = CUDA, train = True):
        
        super().__init__()

        self.databank = databank
        self.index = index

        self.batch_size = batch_size
        self.n_samples = self.index.shape[1]
        self.n_members = self.databank.n_members
        self.train = train

        self.size = int(self.n_samples/batch_size) + (1 if (self.n_samples % batch_size) > 0 else 0)

        lead_time = self.databank.X.shape[2]

        # [batch_size, lead_time, n_members]
        self.X_batch = torch.zeros((batch_size, lead_time, self.n_members), dtype = torch.float32)
        # [batch_size, n_predictors]
        self.P_batch = torch.zeros((batch_size, self.databank.P.shape[1]))
        # [batch_size, lead_time]
        self.Y_batch = torch.zeros((batch_size, lead_time), dtype = torch.float32)

        if cuda:

            self.X_batch = self.X_batch.cuda()
            self.Y_batch = self.Y_batch.cuda()
            self.P_batch = self.P_batch.cuda()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
       
        i0 = idx*self.batch_size
        i1 = min((idx + 1)*self.batch_size, self.n_samples)

        i = self.index[:, i0:i1]
        t = i1 - i0
        
        self.X_batch[:t, :, :] = self.databank.X[i[0], i[1], :, :]
        self.Y_batch[:t, :]    = self.databank.Y[i[0], i[1], :]
        self.P_batch[:t, :]    = self.databank.P[i[0], :]

        if self.train:

            r_m = np.random.permutation(self.n_members)
            X = self.X_batch[:, :, r_m]

        else:
            X = self.X_batch

        return X[:t], self.P_batch[:t], self.Y_batch[:t], i
    
    def __on_epoch_end__(self):
        
        idx = np.random.permutation(self.n_samples)

        self.index[0, :] = self.index[0, idx]
        self.index[1, :] = self.index[1, idx]

def norm(T, remove_nan = False):

    if remove_nan:
        m = T[~np.isnan(T)].mean()
        s = T[~np.isnan(T)].std()
    else:
        m = T.mean()
        s = T.std()

    return (T - m)/s, m, s

def normalize(T, m, s):
    return (T - m)/s

def denormalize(T, m, s):
    return T*s + m

def load_training_dataset(path):

    X = xr.open_dataarray(join(path, "ESSD_benchmark_training_data_forecasts.nc"))
    Y = xr.open_dataarray(join(path, "ESSD_benchmark_training_data_observations.nc")).to_numpy()

    time = [str(x).split("T")[0] for x in X.coords["time"].to_numpy()]
    stations = [str(x) for x in X.coords["station_name"].to_numpy()]

    P_alt_md = np.expand_dims(X.coords["model_altitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lat_md = np.expand_dims(X.coords["model_latitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lon_md = np.expand_dims(X.coords["model_longitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lnd_md = np.expand_dims(X.coords["model_land_usage"].to_numpy(), axis = 0).astype(np.float32)

    P_alt_st = np.expand_dims(X.coords["station_altitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lat_st = np.expand_dims(X.coords["station_latitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lon_st = np.expand_dims(X.coords["station_longitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lnd_st = np.expand_dims(X.coords["station_land_usage"].to_numpy(), axis = 0).astype(np.float32)

    X = np.squeeze(X.to_numpy())

    assert X.shape[-1] == 11
    print(f"Training dataset shape: {X.shape}")

    X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2],) + X.shape[3:])
    Y = np.reshape(Y, (Y.shape[0], Y.shape[1]*Y.shape[2], Y.shape[3]))

    idx = np.logical_not(np.all(np.all(np.isnan(np.reshape(X, (X.shape[0], X.shape[1], np.prod(X.shape[2:])))), axis = -1), axis = 0))
    X = X[:, idx]
    Y = Y[:, idx]

    P_alt_md, P_alt_mean_md, P_alt_std_md = norm(P_alt_md)
    P_lat_md, P_lat_mean_md, P_lat_std_md = norm(P_lat_md)
    P_lon_md, P_lon_mean_md, P_lon_std_md = norm(P_lon_md)
    P_lnd_md, P_lnd_mean_md, P_lnd_std_md = norm(P_lnd_md)

    P_alt_st, P_alt_mean_st, P_alt_std_st = norm(P_alt_st)
    P_lat_st, P_lat_mean_st, P_lat_std_st = norm(P_lat_st)
    P_lon_st, P_lon_mean_st, P_lon_std_st = norm(P_lon_st)
    P_lnd_st, P_lnd_mean_st, P_lnd_std_st = norm(P_lnd_st)

    X, X_mean, X_std = norm(X)
    Y = (Y - X_mean)/X_std

    return X, Y,\
    P_alt_md, P_lat_md, P_lon_md, P_lnd_md,\
    P_alt_st, P_lat_st, P_lon_st, P_lnd_st,\
    X_mean, X_std,\
    X_mean, X_std,\
    P_alt_mean_md, P_alt_std_md, P_lat_mean_md, P_lat_std_md, P_lon_mean_md, P_lon_std_md, P_lnd_mean_md, P_lnd_std_md,\
    P_alt_mean_st, P_alt_std_st, P_lat_mean_st, P_lat_std_st, P_lon_mean_st, P_lon_std_st, P_lnd_mean_st, P_lnd_std_st,\
    np.array(time), np.array(stations)

def load_validation_dataset(path):
    
    X = xr.open_dataarray(join(path, "ESSD_benchmark_test_data_forecasts.nc"))
    Y = xr.open_dataarray(join(path, "ESSD_benchmark_test_data_observations.nc")).to_numpy()

    time = [str(x).split("T")[0] for x in X.coords["time"].to_numpy()]
    stations = [str(x) for x in X.coords["station_name"].to_numpy()]

    P_alt_md = np.expand_dims(X.coords["model_altitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lat_md = np.expand_dims(X.coords["model_latitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lon_md = np.expand_dims(X.coords["model_longitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lnd_md = np.expand_dims(X.coords["model_land_usage"].to_numpy(), axis = 0).astype(np.float32)

    P_alt_st = np.expand_dims(X.coords["station_altitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lat_st = np.expand_dims(X.coords["station_latitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lon_st = np.expand_dims(X.coords["station_longitude"].to_numpy(), axis = 0).astype(np.float32)
    P_lnd_st = np.expand_dims(X.coords["station_land_usage"].to_numpy(), axis = 0).astype(np.float32)

    X = np.squeeze(X.to_numpy())

    print(f"Test dataset shape: {X.shape}")

    return X, Y,\
    P_alt_md, P_lat_md, P_lon_md, P_lnd_md,\
    P_alt_st, P_lat_st, P_lon_st, P_lnd_st,\
    np.array(time), np.array(stations)



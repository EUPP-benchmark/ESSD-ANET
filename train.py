import numpy as np
import torch

from os.path import exists, join
from datetime import datetime
from os import mkdir
import sys

from dataset import Databank, Dataset, norm, normalize, load_training_dataset, load_validation_dataset, denormalize
from model import Model

# Set to True if training on GPU is wanted #
CUDA = True

#########################################################

if len(sys.argv) == 1:
    print("Usage: python3 <path to data> <optional postfix>")
    sys.exit(2)

PATH = sys.argv[1]
POSTFIX = "_" + sys.argv[2] if len(sys.argv) == 3 else ""

#### Load training data, remove nan ####

X, Y,\
P_alt_md, P_lat_md, P_lon_md, P_lnd_md,\
P_alt_st, P_lat_st, P_lon_st, P_lnd_st,\
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

bank = Databank(X, Y, [P_lat_md, P_lon_md, P_lnd_md, P_alt_md, P_lat_st, P_lon_st, P_lnd_st, P_alt_st], cuda = CUDA)

#########################################################

split_trsh = int(bank.index.shape[1]*0.8)
split_idx = np.random.permutation(bank.index.shape[1])

tr_idx = bank.index[:, split_idx[:split_trsh]]
vl_idx = bank.index[:, split_idx[split_trsh:]]

#########################################################

N_EPOCHS = 3000
LEARNING_RATE = 1e-3
BATCH_SIZE    = 128
TOLERANCE = 20

#########################################################

TIMESTAMP = str(datetime.now()).replace(" ", "_").split(":")[0]
OUTPUT_PREFIX = f"Model_{LEARNING_RATE}_{BATCH_SIZE}_{TIMESTAMP}{POSTFIX}" # CHANGE THIS APPROPRIATELY BEFORE TRAINING #

BASE_PATH = join(".", OUTPUT_PREFIX)

if not exists(BASE_PATH):
    mkdir(BASE_PATH)

#########################################################

D_train = Dataset(bank, tr_idx, batch_size = BATCH_SIZE, cuda = CUDA, train = True)
D_valid = Dataset(bank, vl_idx, batch_size = BATCH_SIZE, cuda = CUDA, train = False)

D_train.__on_epoch_end__()
D_valid.__on_epoch_end__()

print(f"Training: {D_train.index.shape}\nValidation: {D_valid.index.shape}")

#########################################################

M = Model(number_of_predictors = 8, lead_time = 21, output_features = 2, hidden_features = 128)
M = M.cuda() if CUDA else M.cpu()

opt = torch.optim.Adam(M.parameters(), lr = LEARNING_RATE, weight_decay = 1e-9)

best_train_loss = None
best_valid_loss = None

train_losses = np.zeros(N_EPOCHS, dtype = np.float32)
valid_losses = np.zeros(N_EPOCHS, dtype = np.float32)

for e in range(N_EPOCHS):

    train_loss = 0.0
    valid_loss = 0.0

    #### Train an epoch ####

    M.train()

    for i in range(len(D_train)):

        x, p, y, _ = D_train.__getitem__(i)

        loss = M.loss(x, p, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()

    D_train.__on_epoch_end__()

    #### Test an epoch ####
    
    M.eval()

    with torch.no_grad():

        for i in range(len(D_valid)):

            x, p, y, _ = D_valid.__getitem__(i)

            valid_loss += M.loss(x, p, y)

    D_valid.__on_epoch_end__()

    valid_loss.item()

    #### Record best losses and save model

    train_loss /= len(D_train)
    valid_loss /= len(D_valid)

    print(f"{e + 1}/{N_EPOCHS}: TLoss {train_loss} VLoss {valid_loss}")
    
    train_losses[e] = train_loss
    valid_losses[e] = valid_loss

    if best_train_loss is None or train_loss < best_train_loss:
        
        best_train_loss = train_loss
        torch.save(M, join(BASE_PATH, f"Model_train"))
    
    if best_valid_loss is None or valid_loss < best_valid_loss:
        
        best_valid_loss = valid_loss
        torch.save(M, join(BASE_PATH, f"Model_valid")) 

    np.savetxt(join(BASE_PATH, f"Train_loss"), train_losses[0:e + 1])
    np.savetxt(join(BASE_PATH, f"Valid_loss"), valid_losses[0:e + 1])

    #### Early stopping ####

    e_cntr = e + 1
    if e_cntr > TOLERANCE:
        
        min_val_loss = valid_losses[0:e_cntr].min()
        print("Early stopping check: ({:.5f})".format(min_val_loss))

        if not (min_val_loss in valid_losses[e_cntr - TOLERANCE:e_cntr]):

            print(f"Early stopping at epoch {e_cntr}")
            break



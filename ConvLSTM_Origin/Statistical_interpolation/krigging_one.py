import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
import os
import logging  # For logging
import psutil  # For measuring memory usage

# Import pykrige for kriging interpolation
from pykrige.ok import OrdinaryKriging

logging.basicConfig(filename='test_log_loss1_loss2_improved.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Set random seed for reproducibility
seed = 128
rng = np.random.RandomState(seed)


def batch_creator(X, batch_size, dataset_length, s):
    batch_x = list()
    batch_y = list()

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length - timesteps - pred_timesteps, batch_size)
    for i in range(len(batch_mask)):
        offset = batch_mask[i]

        batch_x.append(X[offset: offset + timesteps])
        batch_y.append(X[offset + timesteps: offset + timesteps + pred_timesteps])

    batch_x = np.asarray(batch_x)
    batch_x[:, :, station_map[s]] = 0  # Mask the data at station s

    batch_y = np.asarray(batch_y)
    batch_ymap = np.zeros((batch_size, output_size))
    batch_ymap[:, station_map] = 1.0

    batch_x = batch_x.reshape((batch_size, timesteps, image_size, image_size, in_channel))
    batch_y = batch_y.reshape((batch_size, output_size))

    return batch_x, batch_y, batch_ymap


def get_coordinates(e):
    i = int(e / 32)
    j = e % 32
    return i, j


# Load pollution data
pollution_file = '/home/jinyongyun/ConvLSTM_Origin/antwerp_pollution_filtered_origin.h5'
if os.path.isfile(pollution_file):
    with h5py.File(pollution_file, 'r') as hf:
        X = hf['pollution'][:]
        station_map = hf['station_map'][:]

logging.info(f'stationmap size: {len(station_map)}')
logging.info(f'Data shape: {X.shape}')

# Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X.reshape(X.shape[0] * X.shape[1], 1)).reshape(X.shape[0], X.shape[1])

# Split to train, validate, test sets
train_size = (187) * 24 * 60
X_train, X_test = X[:train_size], X[train_size:]
split_size = train_size - (26) * 24 * 60
X_train, X_val = X_train[:split_size], X_train[split_size:]
print('Training set shape: {}'.format(X_train.shape))
print('Validate set shape: {}'.format(X_val.shape))
print('Test set shape: {}'.format(X_test.shape))

# Training Parameters
timesteps = 1  # timesteps
pred_timesteps = 1  # predict timesteps
batch_size = 128

# Network Parameters
grid_size = 1024
image_size = 32
in_channel = 1
output_size = grid_size

# Initialize variables for testing
total_loss_test = 0
total_elapsed_time = 0
total_memory_usage = 0  # Accumulate memory usage
test_steps = int(X_test.shape[0] / batch_size)
test_steps_total = test_steps * len(station_map)

step_counter = 0

for s in range(len(station_map)):
    print(f"Testing station {station_map[s]}")
    loss_test = 0
    elapsed_time = 0
    memory_usage = 0

    for i in range(test_steps):
        batch_x, batch_y, batch_ymap = batch_creator(X_test, batch_size, X_test.shape[0], s)

        actual_values = []
        predicted_values = []

        start_time = time.time()

        for sample_idx in range(batch_size):
            data_sample = batch_x[sample_idx, 0, :, :, 0]  # Shape: (32, 32)
            data_sample_flat = data_sample.flatten()  # Shape: (1024,)
            data_sample_inv = scaler.inverse_transform(data_sample_flat.reshape(-1, 1)).reshape(32, 32)

            # Get the values at other stations
            other_stations = station_map.copy()
            other_stations = np.delete(other_stations, s)

            x_coords = []
            y_coords = []
            values = []
            for idx in other_stations:
                i_coord, j_coord = get_coordinates(idx)
                x_coords.append(i_coord)
                y_coords.append(j_coord)
                values.append(data_sample_inv[i_coord, j_coord])

            # Perform kriging
            OK = OrdinaryKriging(
                x_coords, y_coords, values,
                variogram_model='linear', verbose=False, enable_plotting=False
            )

            station_i, station_j = get_coordinates(station_map[s])

            z, ss = OK.execute('points', station_i, station_j)

            # Get the actual value
            actual_value = batch_y[sample_idx, station_map[s]]
            # Inverse transform actual value
            actual_value_inv = scaler.inverse_transform([[actual_value]])[0, 0]

            # Store values
            actual_values.append(actual_value_inv)
            predicted_values.append(z[0])

        loss_value = sqrt(mean_squared_error(actual_values, predicted_values))
        elapsed_time += time.time() - start_time

        # Measure memory usage
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
        memory_usage += current_memory

        loss_test += loss_value

        step_counter += 1
        progress = (step_counter / test_steps_total) * 100

    # Output performance for each station
    avg_memory_usage = memory_usage / test_steps  # Average memory usage
    logging.info(f"Test Error = {loss_test / test_steps:.6f}, Elapsed time = {elapsed_time / test_steps:.3f} seconds, Memory Usage = {avg_memory_usage:.2f} MB")
    total_loss_test += loss_test
    total_elapsed_time += elapsed_time
    total_memory_usage += memory_usage

# After testing all stations, output overall average interpolation error (spRMSE)
logging.info(f"total loss test = {total_loss_test}")
logging.info(f"total elapsed time = {total_elapsed_time}")
logging.info(f"spRMSE = {total_loss_test / test_steps_total:.6f}, Elapsed time = {total_elapsed_time / test_steps_total:.3f} seconds, Average Memory Usage = {total_memory_usage / test_steps_total:.2f} MB")

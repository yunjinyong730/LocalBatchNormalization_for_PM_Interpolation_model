import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from keras.layers import ConvLSTM2D, Conv2D, Input, BatchNormalization, Flatten, Dense, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
import os
import logging
import psutil  # For measuring memory usage

# Set up logging
logging.basicConfig(filename='convlstm_improved_weight_batch_normal_2.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Random seed setting
seed = 128
rng = np.random.RandomState(seed)

# Batch creation function
def batch_creator(X, batch_size, dataset_length, s=None):
    batch_x = list()
    batch_y = list()

    batch_mask = rng.choice(dataset_length - timesteps - pred_timesteps, batch_size)
    for i in range(len(batch_mask)):
        offset = batch_mask[i]
        batch_x.append(X[offset : offset + timesteps])
        batch_y.append(X[offset + timesteps : offset + timesteps + pred_timesteps])
    
    batch_x = np.asarray(batch_x)
    if s is not None:
        batch_x[:, :, station_map[s]] = 0  # Set the selected station data to 0 for each test step

    batch_y = np.asarray(batch_y)
    batch_ymap = np.zeros((batch_size, output_size))
    batch_ymap[:, station_map] = 1.0

    batch_x = batch_x.reshape((batch_size, timesteps, image_size, image_size, in_channel))
    batch_y = batch_y.reshape((batch_size, output_size))

    return batch_x, batch_y, batch_ymap

# Load the h5 dataset
pollution_file = '/home/jinyongyun/ConvLSTM_Origin/antwerp_pollution_filtered_origin.h5'
if os.path.isfile(pollution_file):
    with h5py.File(pollution_file, 'r') as hf:
        X = hf['pollution'][:]
        station_map = hf['station_map'][:]

# Normalizing the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X.reshape(X.shape[0]*X.shape[1],1)).reshape(X.shape[0], X.shape[1])

# Train/Validation/Test split
train_size = (187)*24*60
X_train, X_test = X[:train_size], X[train_size:]
split_size = train_size - (26)*24*60
X_train, X_val = X_train[:split_size], X_train[split_size:]

# Training Parameters
timesteps = 1  # Number of timesteps
pred_timesteps = 1  # Prediction timesteps
learning_rate = 0.001
training_steps = 1200
batch_size = 128
display_step = 20
is_training = True

# Network Parameters
grid_size = 1024
image_size = 32
in_channel = 1
out_channel = [64]
fc_size = 1000
dr_rate = 0.5
loss_ratio = grid_size / len(station_map)
output_size = grid_size


### 수정된 Regional Batch Normalization Implementation
def create_regional_batch_norm_layer(inputs, grid_size, out_channel):
    # 각 그리드 포인트마다 BatchNormalization을 독립적으로 적용
    regions = []
    for i in range(grid_size):
        row = i // 32
        col = i % 32
        # 슬라이싱 시 각 포인트의 한 채널을 가져오기 위해 채널 축을 유지한 채 슬라이싱 수행
        region_input = inputs[:, row:row+1, col:col+1, :]  # (batch_size, 1, 1, filters)
        region_norm = BatchNormalization()(region_input)
        regions.append(region_norm)
    
    # Concatenate를 통해 모든 지역의 BatchNormalization 결과를 합침
    normalized_output = Concatenate(axis=1)(regions)  # (batch_size, grid_size, 1, filters)
    
    # 형상을 맞추기 위해 다시 (batch_size, 32, 32, filters)로 재구성
    normalized_output = tf.reshape(normalized_output, (-1, 32, 32, out_channel))
    return normalized_output

# Find neighbors for each grid point
def find_neighbor(e):
    i = int(e / 32)
    j = e - i * 32
    if i == 0:
        i_nei = [0, 1]
    elif i == 31:
        i_nei = [i - 1, i]
    else:
        i_nei = [i - 1, i, i + 1]
    if j == 0:
        j_nei = [0, 1]
    elif j == 31:
        j_nei = [j - 1, j]
    else:
        j_nei = [j - 1, j + 1]
    e_nei = list()
    for t in range(len(i_nei)):
        for k in range(len(j_nei)):
            nei_idx = i_nei[t] * 32 + j_nei[k]
            if nei_idx != e:
                e_nei.append(nei_idx)
    return e_nei

# Create a dictionary of neighbors for each grid point
def neighbors():
    my_dict = {}
    for i in range(grid_size):
        nei = find_neighbor(i)
        my_dict[i] = nei
    return my_dict

neighbors_dict = neighbors()

# ConvLSTM2D Model Creation with Regional Batch Normalization
def create_conv_lstm_model_with_regional_bn(timesteps, image_size, in_channel, out_channel, grid_size):
    input_layer = Input(shape=(timesteps, image_size, image_size, in_channel))
    
    # ConvLSTM Layer with Regional Batch Normalization
    conv_lstm = ConvLSTM2D(
        filters=out_channel[0],
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False
    )(input_layer)
    
    # 지역별 Batch Normalization 적용
    regional_bn_output = create_regional_batch_norm_layer(conv_lstm, grid_size, out_channel[0])
    
    # 1x1 Convolutional Layer for output
    conv_output = Conv2D(filters=pred_timesteps, kernel_size=(1, 1), activation='sigmoid')(regional_bn_output)
    
    # Flatten the output
    flat_output = Flatten()(conv_output)
    
    # Fully Connected Layer with Dropout
    dense_output = Dense(fc_size, activation='relu')(flat_output)
    dense_output = Dropout(dr_rate)(dense_output)
    
    # Final Output Layer
    final_output = Dense(output_size, activation='linear')(dense_output)
    
    # Create Model
    model = Model(inputs=input_layer, outputs=final_output)
    
    return model

# Create the ConvLSTM model
model = create_conv_lstm_model_with_regional_bn(timesteps, image_size, in_channel, out_channel, grid_size)

# Custom loss functions
def compute_loss1(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)) * loss_ratio)

def compute_loss2(y_true, y_pred, neighbors_dict):
    loss2 = 0.0
    for e in range(grid_size):
        e_neighbors = neighbors_dict[e]
        for nei in e_neighbors:
            loss2 += tf.reduce_mean(tf.square(y_pred[:, e] - y_true[:, nei]))
    return loss2 / grid_size

def total_loss(y_true, y_pred, neighbors_dict):
    loss1 = compute_loss1(y_true, y_pred)
    loss2 = compute_loss2(y_true, y_pred, neighbors_dict)
    return 2 * loss1 + loss2


neighbors_dict = neighbors()


# Compile model with custom loss
model.compile(
    optimizer=Adam(learning_rate),
    loss=lambda y_true, y_pred: total_loss(y_true, y_pred, neighbors_dict)
)


# Load pre-trained model weights if they exist
model_path = "convlstm_model_regional_bn_weights.h5"
if os.path.isfile(model_path):
    model.load_weights(model_path) 
    logging.info("Model loaded from file.")

# Testing Loop with Enhanced Reporting and Memory Measurement
total_loss_test = 0
total_elapsed_time = 0
total_memory_usage = 0  # Accumulate memory usage
batch_size = 64
test_steps = int(X_test.shape[0] / batch_size)

for s in range(len(station_map)):
    print(f"Testing station {station_map[s]}")
    loss_test = 0
    elapsed_time = 0
    memory_usage = 0

    for i in range(test_steps):
        batch_x, batch_y, batch_ymap = batch_creator(X_test, batch_size, X_test.shape[0], s)
            
        start_time = time.time()
        pred_test = model.predict(batch_x)
            
        # Measure memory usage
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / (1024 ** 2)  # Current memory usage in MB
        memory_usage += current_memory
            
        # Inverse transform to get actual PM2.5 values
        inv_out = scaler.inverse_transform(pred_test.flatten().reshape(-1, 1))
        inv_yhat = inv_out[station_map[s], :]
        inv_y = scaler.inverse_transform(batch_y.flatten().reshape(-1, 1))
        inv_y = inv_y[station_map[s], :]
            
        # RMSE between interpolated and actual values
        loss_value = sqrt(mean_squared_error(inv_y, inv_yhat))
            
        elapsed_time += time.time() - start_time
        loss_test += loss_value
        
    avg_memory_usage = memory_usage / test_steps  # Calculate average memory usage
    logging.info(f"Test Error = {loss_test / test_steps:.6f}, Elapsed time = {elapsed_time / test_steps:.3f} seconds")
    total_loss_test += loss_test
    total_elapsed_time += elapsed_time
    total_memory_usage += memory_usage

# Print overall interpolation error for all stations
test_steps_total = test_steps * len(station_map)
logging.info(f"Total loss test = {total_loss_test}")
logging.info(f"Total elapsed time = {total_elapsed_time}")
logging.info(f"spRMSE = {total_loss_test / test_steps_total:.6f}, Elapsed time = {total_elapsed_time / test_steps_total:.3f} seconds, Average Memory Usage = {total_memory_usage / test_steps_total:.2f} MB")

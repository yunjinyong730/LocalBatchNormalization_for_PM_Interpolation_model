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

# Random seed setting
seed = 128
rng = np.random.RandomState(seed)

# Batch creation function
def batch_creator(X, batch_size, dataset_length):
    batch_x = list()
    batch_y = list()

    batch_mask = rng.choice(dataset_length - timesteps - pred_timesteps, batch_size)
    for i in range(len(batch_mask)):
        offset = batch_mask[i]
        batch_x.append(X[offset : offset + timesteps])
        batch_y.append(X[offset + timesteps : offset + timesteps + pred_timesteps])
    
    batch_x = np.asarray(batch_x)
    batch_y = np.asarray(batch_y)
    batch_ymap = np.zeros((batch_size, output_size))
    batch_ymap[:, station_map] = 1.0

    batch_x = batch_x.reshape((batch_size, timesteps, image_size, image_size, in_channel))
    batch_y = batch_y.reshape((batch_size, output_size))

    return batch_x, batch_y, batch_ymap


def find_neighbor(e):
    i = int(e / 32)
    j = e % 32  # j 값을 i와 마찬가지로 계산

    # i 값에 따른 이웃 설정
    if i == 0:
        i_nei = [0, 1]
    elif i == 31:
        i_nei = [30, 31]
    else:
        i_nei = [i - 1, i, i + 1]

    # j 값에 따른 이웃 설정
    if j == 0:
        j_nei = [0, 1]
    elif j == 31:
        j_nei = [30, 31]
    else:
        j_nei = [j - 1, j, j + 1]

    # 이웃 인덱스를 생성
    e_nei = []
    for t in i_nei:
        for k in j_nei:
            nei_idx = t * 32 + k
            if nei_idx != e:
                e_nei.append(nei_idx)

    return e_nei


# 전체 그리드에서 각 점에 대해 이웃을 찾아 딕셔너리롤 반환
def neighbors():
    my_dict = {}
    for i in range(grid_size):
        nei = find_neighbor(i)
        my_dict[i] = nei
    return my_dict

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
training_steps = 120
batch_size = 64
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
    
    
    # Final Output Layer
    final_output = Dense(output_size, activation='linear')(flat_output)
    
    # Create Model
    model = Model(inputs=input_layer, outputs=final_output)
    
    return model

# Create the ConvLSTM model with regional batch normalization
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

# Training and Evaluation
if is_training:
    for step in range(1, training_steps + 1):
        batch_x, batch_y, batch_ymap = batch_creator(X_train, batch_size, X_train.shape[0])
        history = model.fit(batch_x, batch_y, batch_size=batch_size, verbose=0)
        
        # Validation
        batch_x_val, batch_y_val, batch_ymap_val = batch_creator(X_val, batch_size, X_val.shape[0])
        val_loss = model.evaluate(batch_x_val, batch_y_val, verbose=0)
        
        if step % display_step == 0 or step == 1:
            print(f"Step {step}, Train Loss: {history.history['loss'][0]}, Validate Loss: {val_loss}")

    # Save the model
    model.save('convlstm_model_regional_bn.h5')
    model.save_weights('convlstm_model_regional_bn_weights.h5')
    print("Model saved.")

# Testing
if not is_training:
    model = tf.keras.models.load_model('convlstm_model_regional_bn.h5', custom_objects={'total_loss': total_loss})
    loss_test = 0
    elapsed_time = 0
    test_steps = int(X_test.shape[0] / batch_size)
    
    for i in range(test_steps):
        batch_x_test, batch_y_test, batch_ymap_test = batch_creator(X_test, batch_size, X_test.shape[0])
        
        start_time = time.time()
        pred = model.predict(batch_x_test)
        
        # Inverse transform to get actual PM2.5 values
        inv_pred = scaler.inverse_transform(pred.flatten().reshape(-1, 1))
        inv_yhat = np.multiply(inv_pred, batch_ymap_test.flatten().reshape(-1, 1))
        inv_y = scaler.inverse_transform(batch_y_test.flatten().reshape(-1, 1))
        loss_value = sqrt(mean_squared_error(inv_y, inv_yhat) * loss_ratio)
        
        elapsed_time += time.time() - start_time
        loss_test += loss_value
    
    # Print final test error
    print(f"Test Error: {loss_test / test_steps:.6f}, Elapsed time per step: {elapsed_time / test_steps:.3f}")
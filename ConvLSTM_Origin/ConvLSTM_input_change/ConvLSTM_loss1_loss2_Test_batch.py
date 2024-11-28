import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from keras.layers import ConvLSTM2D, Conv2D
from keras.models import Model, load_model
from keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
import os
import logging  # 로그를 위해 추가


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


logging.basicConfig(filename='test_log_tflite_improved_batch.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# 시드값 고정
seed = 128
rng = np.random.RandomState(seed)

# Sequential Batch 생성 함수
# 개선된 배치 생성 함수: 실제 배치 크기에 맞춰 동적으로 처리

def batch_creator_sequential(X, batch_size, dataset_length, start_index=0, s=None):
    batch_x = list()
    batch_y = list()

    for i in range(batch_size):
        offset = start_index + i * (timesteps + pred_timesteps)
        if offset + timesteps + pred_timesteps >= dataset_length:
            # 현재 배치에 더 이상 추가하지 않도록 처리
            break

        batch_x.append(X[offset : offset + timesteps])
        batch_y.append(X[offset + timesteps : offset + timesteps + pred_timesteps])

    # 실제 배치 크기를 계산
    actual_batch_size = len(batch_x)

    # 넘파이 배열로 변환
    batch_x = np.asarray(batch_x)
    batch_y = np.asarray(batch_y)

    # batch_ymap 배열 생성
    batch_ymap = np.zeros((actual_batch_size, output_size))
    if s is not None:
        batch_x[:, :, station_map[s]] = 0  # 각 테스트 스텝마다 선택된 측정소 데이터 0으로 설정
    
    batch_ymap[:, station_map] = 1.0

    # 실제 배치 크기에 맞게 재구성
    batch_x = batch_x.reshape((actual_batch_size, timesteps, image_size, image_size, in_channel))
    batch_y = batch_y.reshape((actual_batch_size, output_size))

    return batch_x, batch_y, batch_ymap

# 이웃 찾기 함수 개선
def find_neighbor_optimized(e):
    i, j = divmod(e, 32)
    i_nei = range(max(0, i - 1), min(32, i + 2))
    j_nei = range(max(0, j - 1), min(32, j + 2))
    return [x * 32 + y for x in i_nei for y in j_nei if x * 32 + y != e]

# 전체 그리드에서 각 점에 대해 이웃을 찾아 딕셔너리로 반환
def neighbors_optimized():
    return {i: find_neighbor_optimized(i) for i in range(grid_size)}

# Load pollution data
pollution_file = '/home/jinyongyun/ConvLSTM_Origin/antwerp_pollution_filtered_origin.h5'
if os.path.isfile(pollution_file):
    with h5py.File(pollution_file, 'r') as hf:
        X = hf['pollution'][:]
        station_map = hf['station_map'][:]

logging.info(f'stationmap size: {len(station_map)}')
logging.info(f'Data shape: {X.shape}')

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X.reshape(X.shape[0]*X.shape[1],1)).reshape(X.shape[0], X.shape[1])

# Split to train, validate, test set 나누기
train_size = (187) * 24 * 60
X_train, X_test = X[:train_size], X[train_size:]
split_size = train_size - (26) * 24 * 60
X_train, X_val = X_train[:split_size], X_train[split_size:]
print('Training set shape: {}'.format(X_train.shape))
print('Validate set shape: {}'.format(X_val.shape))
print('Test set shape: {}'.format(X_test.shape))

# Training Parameters
timesteps = 5  # 개선된 timesteps
pred_timesteps = 1  # predict timesteps
learning_rate = 0.001
training_steps = 80
batch_size = 128
display_step = 20
is_training = False

# Network Parameters
grid_size = 1024
image_size = 32
in_channel = 1
out_channel = [64, 32]  # 두 개의 ConvLSTM 계층을 위한 필터
fc_size = 1000
dr_rate = 0.5
loss_ratio = grid_size / len(station_map)
output_size = grid_size

# 개선된 ConvLSTM 모델 생성
def create_deep_conv_lstm_model(timesteps, image_size, in_channel, out_channel):
    model = tf.keras.Sequential()
    
    # 첫 번째 ConvLSTM Layer
    model.add(tf.keras.layers.ConvLSTM2D(
        filters=out_channel[0],
        kernel_size=(3, 3),
        input_shape=(timesteps, image_size, image_size, in_channel),
        padding='same',
        return_sequences=True))
    
    # 두 번째 ConvLSTM Layer
    model.add(tf.keras.layers.ConvLSTM2D(
        filters=out_channel[1],
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False))

    # Additional Conv2D Layer
    model.add(tf.keras.layers.Conv2D(filters=pred_timesteps, kernel_size=(1, 1), activation='sigmoid'))

    # Flatten the output
    model.add(tf.keras.layers.Flatten())

    return model

# Create the ConvLSTM model with improved architecture
model = create_deep_conv_lstm_model(timesteps, image_size, in_channel, out_channel)

# loss1 계산 함수
def compute_loss1(y_true, y_pred):
    return tf.sqrt(tf.losses.mean_squared_error(y_true, y_pred) * loss_ratio)

# loss2 계산 함수
def compute_loss2(y_true, y_pred, neighbors_dict):
    loss2 = 0.0
    for e in range(grid_size):
        e_neighbors = neighbors_dict[e]
        for nei in e_neighbors:
            loss2 += tf.reduce_mean(tf.square(y_pred[:, e] - y_true[:, nei]))
    return loss2 / grid_size

# 모델 컴파일: loss1 * 2 + loss2 사용
def total_loss(y_true, y_pred, neighbors_dict):
    loss1 = compute_loss1(y_true, y_pred)
    loss2 = compute_loss2(y_true, y_pred, neighbors_dict)
    return 2 * loss1 + loss2

neighbors_dict = neighbors_optimized()

# 커스텀 손실 함수를 사용하는 모델 컴파일
def compile_model(model, neighbors_dict):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=lambda y_true, y_pred: total_loss(y_true, y_pred, neighbors_dict)
    )

compile_model(model, neighbors_dict)

# Load pre-trained model weights if they exist
model_path = "convlstm_model_improved_sequential_weight.h5"
if os.path.isfile(model_path):
    model.load_weights(model_path)
    logging.info("Model loaded from file.")

# Test the model
total_loss_test = 0
total_elapsed_time = 0
test_steps = int(X_test.shape[0] / batch_size)

for s in range(len(station_map)):
    print(f"Testing station {station_map[s]}")
    loss_test = 0
    elapsed_time = 0
    for i in range(test_steps):
        start_idx = i * batch_size
        batch_x, batch_y, batch_ymap = batch_creator_sequential(X_test, batch_size, X_test.shape[0], start_idx, s)

        start_time = time.time()

        # 보간된 값을 얻고
        pred_test = model.predict(batch_x)

        # 역정규화
        inv_out = scaler.inverse_transform(pred_test.flatten().reshape(-1, 1))

        # 보간값
        inv_yhat = inv_out[station_map[s], :]

        # 실제값
        inv_y = scaler.inverse_transform(batch_y.flatten().reshape(-1, 1))
        inv_y = inv_y[station_map[s], :]

        # 보간값과 실제값의 RMSE
        loss_value = sqrt(mean_squared_error(inv_y, inv_yhat))

        elapsed_time += time.time() - start_time
        loss_test += loss_value

    logging.info(f"Test Error = {loss_test / test_steps:.6f}, Elapsed time = {elapsed_time / test_steps:.3f} seconds")
    total_loss_test += loss_test
    total_elapsed_time += elapsed_time

# 모든 측정소에 대한 평균 보간 오차 및 시간
test_steps_total = test_steps * len(station_map)
logging.info(f"total loss test = {total_loss_test}")
logging.info(f"total elapsed time = {total_elapsed_time}")
logging.info(f"spRMSE = {total_loss_test / test_steps_total:.6f}, Elapsed time = {total_elapsed_time / test_steps_total:.3f} seconds")
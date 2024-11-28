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
import psutil  # 메모리 사용량 측정을 위해 추가


logging.basicConfig(filename='test_log_tflite_new1.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# 시드값 고정
seed = 128
rng = np.random.RandomState(seed)

def batch_creator(X, batch_size, dataset_length, s):
    batch_x = list()
    batch_y = list()

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length - timesteps - pred_timesteps, batch_size)
    for i in range(len(batch_mask)):
        offset = batch_mask[i]
        batch_x.append(X[offset : offset + timesteps])
        batch_y.append(X[offset + timesteps : offset + timesteps + pred_timesteps])
    
    batch_x = np.asarray(batch_x)
    batch_x[:, :, station_map[s]] = 0 # 각 테스트 스텝마다 선택된 측정소 데이터 0으로 설정

    batch_y = np.asarray(batch_y)
    batch_ymap = np.zeros((batch_size, output_size))
    batch_ymap[:, station_map] = 1.0

    batch_x = batch_x.reshape((batch_size, timesteps, image_size, image_size, in_channel))
    batch_y = batch_y.reshape((batch_size, output_size))

    return batch_x, batch_y, batch_ymap


# 그리드에서 특정 점의 이웃을 찾기
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

# 전체 그리드에서 각 점에 대해 이웃을 찾아 딕셔너리롤 반환
def neighbors():
    my_dict = {}
    for i in range(grid_size):
        nei = find_neighbor(i)
        my_dict[i] = nei
    return my_dict

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
train_size = (187)*24*60
X_train, X_test = X[:train_size], X[train_size:]
split_size = train_size - (26)*24*60
X_train, X_val = X_train[:split_size], X_train[split_size:]
logging.info(f'Training set shape: {X_train.shape}')
logging.info(f'Validate set shape: {X_val.shape}')
logging.info(f'Test set shape: {X_test.shape}')

# Training Parameters
timesteps = 1 # timesteps
pred_timesteps = 1 # predict timesteps
learning_rate = 0.001
training_steps = 1200
batch_size = 128
display_step = 20
is_training = False

# Network Parameters
grid_size = 1024
image_size = 32
in_channel = 1
out_channel = [64]
fc_size = 1000
dr_rate = 0.5
loss_ratio = grid_size / len(station_map)
output_size = grid_size

### Keras-based ConvLSTM Model
# ConvLSTM2D 모델 생성
def create_conv_lstm_model(timesteps, image_size, in_channel, out_channel):
    model = tf.keras.Sequential()
    # ConvLSTM Layer
    model.add(tf.keras.layers.ConvLSTM2D(
        filters=out_channel[0],
        kernel_size=(3, 3),
        input_shape=(timesteps, image_size, image_size, in_channel),
        padding='same',
        return_sequences=False))
    
    # 1x1 Convolutional Layer for output
    model.add(tf.keras.layers.Conv2D(filters=pred_timesteps, kernel_size=(1, 1), activation='sigmoid'))
    
    # Flatten the output
    model.add(tf.keras.layers.Flatten())
    
    return model

# Create the ConvLSTM model
model = create_conv_lstm_model(timesteps, image_size, in_channel, out_channel)

# loss1 계산 함수: 실제 미세먼지 값과 예측 값의 차이를 최소화하는 손실 함수
def compute_loss1(y_true, y_pred):
    return tf.sqrt(tf.losses.mean_squared_error(y_true, y_pred) * loss_ratio)


# loss2 계산 함수: 인접 station들 간의 예측 값과 실제 미세먼지 값의 차이를 최소화하는 손실 함수
def compute_loss2(y_true, y_pred, neighbors_dict):
    loss2 = 0.0
    for e in range(grid_size):
        e_neighbors = neighbors_dict[e]
        for nei in e_neighbors:
            # 이웃 station의 실제값과 현재 station의 예측값 차이 계산
            loss2 += tf.reduce_mean(tf.square(y_pred[:, e] - y_true[:, nei]))
    return loss2 / grid_size  # 전체 평균

# 모델 컴파일: loss1 * 2 + loss2 사용
def total_loss(y_true, y_pred, neighbors_dict):
    loss1 = compute_loss1(y_true, y_pred)
    loss2 = compute_loss2(y_true, y_pred, neighbors_dict)
    return 2 * loss1 + loss2

neighbors_dict = neighbors()

# 커스텀 손실 함수를 사용하는 모델 컴파일
def compile_model(model, neighbors_dict):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=lambda y_true, y_pred: total_loss(y_true, y_pred, neighbors_dict)
    )

compile_model(model, neighbors_dict)

# Load pre-trained model weights if they exist
model_path = "convlstm_model_improved_weight.h5"
if os.path.isfile(model_path):
    model.load_weights(model_path) 
    logging.info("Model loaded from file.")

def convert_to_tflite(model, quantize=False):
    # 모델을 TFLite로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Select TF Ops를 사용하여 지원하지 않는 연산을 포함하도록 설정
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite 기본 연산
        tf.lite.OpsSet.SELECT_TF_OPS     # TensorFlow 연산 포함
    ]
    
    # TensorList와 같은 동적 연산을 변환하지 않도록 설정
    converter._experimental_lower_tensor_list_ops = False

    if quantize:
        # 포스트 트레이닝 양자화 적용 (전체 int8 양자화)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # float16 양자화 적용
    
    tflite_model = converter.convert()

    # 변환된 TFLite 모델을 저장
    tflite_model_file = 'convlstm_model_loss1_loss2.tflite'
    with open(tflite_model_file, 'wb') as f:
        f.write(tflite_model)

    logging.info(f"TFLite model saved at {tflite_model_file}")

# TFLite 양자화된 모델 생성
convert_to_tflite(model, quantize=False)

# Load TFLite model and allocate tensors.
def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_with_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure the input data is in FLOAT32 format as required by TFLite
    input_data = input_data.astype(np.float32)

    # Get the expected input shape from the TFLite model
    input_shape = input_details[0]['shape']

    # Log to inspect the input shape and input data shape
    #logging.info(f"Expected input shape: {input_shape}, Actual input shape: {input_data.shape}")

    # Reshape input_data to match the expected shape
    input_data = input_data.reshape(input_shape)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


# Load the TFLite model
tflite_model_path = 'convlstm_model_loss1_loss2.tflite'
interpreter = load_tflite_model(tflite_model_path)

# Test the TFLite model with batch size of 1
total_loss_test = 0
total_elapsed_time = 0
total_memory_usage = 0  # 메모리 사용량 누적 합산



# Set batch size to 1 for TFLite model
batch_size = 1  
test_steps = X_test.shape[0]
test_steps_total = test_steps * len(station_map)

# Progress tracking
step_counter = 0

for s in range(len(station_map)):
    logging.info(f"Testing station {station_map[s]}")
    loss_test = 0
    elapsed_time = 0
    memory_usage = 0

    for i in range(test_steps):
        batch_x, batch_y, batch_ymap = batch_creator(X_test, batch_size, X_test.shape[0], s)

        start_time = time.time()

        # Predict using the TFLite model
        pred_test = predict_with_tflite(interpreter, batch_x)


        # 메모리 사용량 측정
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / (1024 ** 2)  # 현재 메모리 사용량 (MB 단위)
        memory_usage += current_memory
        
        
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
        

        # 진행 상황 업데이트
        step_counter += 1
        progress = (step_counter / test_steps_total) * 100
        #logging.info(f"Progress: {progress:.2f}%")

    # 각 측정소에 대한 보간 성능을 출력
    avg_memory_usage = memory_usage / test_steps  # 평균 메모리 사용량 계산
    logging.info(f"Test Error = {loss_test / test_steps:.6f}, Elapsed time = {elapsed_time / test_steps:.3f} seconds")
    total_loss_test += loss_test
    total_elapsed_time += elapsed_time
    total_memory_usage += memory_usage


# 모든 측정소에 대한 보간 테스트 완료 후, 전체적인 평균 보간 오차(spRMSE)를 출력
test_steps_total = test_steps * len(station_map)
logging.info(f"total loss test = {total_loss_test}")
logging.info(f"total elapsed time = {total_elapsed_time}")
logging.info(f"spRMSE = {total_loss_test / test_steps_total:.6f}, Elapsed time = {total_elapsed_time / test_steps_total:.3f} seconds, Average Memory Usage = {total_memory_usage / test_steps_total:.2f} MB")

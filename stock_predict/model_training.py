import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import os

# 1. 데이터 수집
def download_stock_data(ticker):
    data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
    return data

# 2. 데이터 전처리
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


# 3. 모델 학습
def build_and_train_model(X_train, y_train, X_test, y_test, time_step, epochs=5, batch_size=1):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # 모델 저장 - 네이티브 Keras 포맷으로 저장
    model_path = "lstm_model"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(os.path.join(model_path, "model"))

    return model, train_predict, test_predict

# 4. 예측 및 시각화
def plot_predictions(data, train_predict, test_predict, time_step, scaler):
    # 예측값 역변환
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # 원본 데이터 역변환
    original_data = scaler.inverse_transform(data)

    # 그래프 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(original_data, label="Original Data")
    
    # 학습 데이터의 예측 결과
    train_range = np.arange(time_step, time_step + len(train_predict))
    plt.plot(train_range, train_predict, label="Train Predict")

    # 테스트 데이터의 예측 결과
    test_range = np.arange(time_step + len(train_predict), time_step + len(train_predict) + len(test_predict))
    plt.plot(test_range, test_predict, label="Test Predict")

    plt.legend()
    plt.show()

# 메인 실행
if __name__ == "__main__":
    ticker = "AAPL"
    data = download_stock_data(ticker)
    processed_data, scaler = preprocess_data(data)
    
    time_step = 50
    X, Y = create_dataset(processed_data, time_step)
    
    # 데이터셋 분리
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = Y[0:train_size], Y[train_size:len(Y)]
    
    # 입력 데이터 형태 변환
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model, train_predict, test_predict = build_and_train_model(X_train, y_train, X_test, y_test, time_step)

    plot_predictions(processed_data, train_predict, test_predict, time_step, scaler)

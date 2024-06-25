from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf

app = Flask(__name__)

# 모델 로드
model = load_model('lstm_model.h5')  # 모델 파일 경로에 맞게 수정

# 스케일러 초기화
scaler = MinMaxScaler(feature_range=(0, 1))

# 데이터 수집 함수
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# 데이터 전처리 함수
def preprocess_data(data):
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data

# 예측 함수
def make_predictions(data, time_step=50):
    scaled_data = preprocess_data(data)
    X = []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i, 0])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# 예측 API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data['ticker']
    start_date = data['start_date']
    end_date = data['end_date']
    
    stock_data = download_stock_data(ticker, start_date, end_date)
    predictions = make_predictions(stock_data)
    
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)

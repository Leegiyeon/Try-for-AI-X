# AI

> 1. 문장에 대한 감정 분석 모델(test)<br>
> Model: Pipeline('sentiment-analysis')<br>

> 2. 주식 데이터 수집, 분석, 예측 모델(stock_predict)<br>
  > (1) data_collection.py를 실행하여 데이터 수집<br>
  > (2) model_training.py를 실행하여 예측 모델 학습(LSTM)<br>
  > (3) app.py를 실행하여 학습된 모델 로드<br>
  > (4) client.py를 실행하여 해당 요청으로 실제 데이터와 비교 분석

> 3. 인공지능 시인(ai_poet) with langchain<br>
  > (1) OPEN AI를 활용한 시 작성 with Streamlit<br>
  > (2) Deploy URL: [App](https://ai-poet-test-deploy.streamlit.app)
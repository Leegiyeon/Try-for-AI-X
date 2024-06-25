import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {
    'ticker': 'AAPL',
    'start_date': '2023-01-01',
    'end_date': '2024-01-01'
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    predictions = response.json()
    print(predictions)
else:
    print(f"Error: {response.status_code}, {response.text}")

import requests

url = "http://127.0.0.1:5000/predict"
data = {"date": "2023-01-10"}

response = requests.post(url, json=data)
print(response.json())

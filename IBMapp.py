# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:11:40 2020

@author: rincy
"""

from flask import Flask, request, render_template
import os
import numpy as np
import pandas as pd
import pickle
import requests
import json

app = Flask(__name__)
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "1yXp97OYKZQ8Ogy-ZD7ccFWHqNTT0yRd5euzqtpRC1Wv"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]
header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
model = pickle.load(open('PAE_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("indexEA.html")


@app.route('/predict', methods=["POST", "GET"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    print(features_value)

    features_name = ['satisfaction_level', 'last_evaluation', 'number_project',
                     'average_montly_hours', 'time_spend_company', 'Work_accident',
                     'promotion_last_5years', 'department', 'salary']

    scaler = pickle.load(open("scaler.pkl", "rb"))
    X_test_scaled = scaler.transform(features_value)
    prediction = model.predict(X_test_scaled)
    output = prediction[0]

    return render_template('resultEA.html', prediction_text=output)
    payload_scoring = {"input_data": [{"field": ['satisfaction_level', 'last_evaluation', 'number_project',
                                                 'average_montly_hours', 'time_spend_company', 'Work_accident',
                                                 'promotion_last_5years', 'department', 'salary'],
                                       "values": [[input_features]]}]}
    response_scoring = requests.post(
        'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/fbd61a8a-69cc-49fe-90c5-a2e7619a2ac5/predictions?version=2022-07-31',
        json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    predictions = response_scoring.json()
    print(predictions['predictions'][0]['values'][0][0])

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
    # http_server = WSGIServer(('0.0.0.0', port), app)
    # http_server.serve_forever()
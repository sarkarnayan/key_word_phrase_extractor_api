# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:30:56 2019

@author: Nayan
"""

# Import libraries
from api_model import model
from flask import Flask, request, jsonify

app = Flask(__name__)# Load the model
model = model()
@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(data['input_data'])    # Take the first value of prediction
    output = prediction    
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
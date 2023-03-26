import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()


# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


def model_predict(img, model):
    food_list = ['apple_pie','pizza','omelette', 'samosa']
    img = img.resize((224, 224))
    x = image.img_to_array(img)                    
    x = np.expand_dims(x, axis=0)         
    x /= 255. 
    pred = model.predict(x)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]
    return pred_value


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)
        'apple_pie','pizza','omelette', 'samosa'
        dict1 = {'apple_pie': {'name': "Apple Pie\n", 'calories': "237 Kcal/100g", 'healthy': "No"},
             'pizza': {'name': "Pizza\n", 'calories': "266 Kcal/100g", 'healthy': "No"},
              'omelette': {'name': "Omelette\n", 'calories': "154 Kcal/100g", 'healthy': "Yes"},
               'samosa': {'name': "Samosa\n", 'calories': "262 Kcal/100g", 'healthy': "No"}}
        # Serialize the result, you can add additional fields
        return jsonify(result=str(dict1[preds]))

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

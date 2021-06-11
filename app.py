#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import jsonify
import pandas as pd
import pickle
import datetime
import numpy as np
import calendar
import os
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input


# In[2]:


if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the model
#model = load_model('resnet152V2_cotton_disease_model.h5',compile=False)


# In[3]:


def predict_disease(file_path, model):
    img = image.load_img(file_path, target_size=(224,224,3))
    X = image.img_to_array(img)
    X = X/255
    X = np.expand_dims(X, axis=0)
    X = preprocess_input(X)
    
#    pred = model.predict(X)
#    pred = np.argmax(pred, axis=1)
    pred=2
    print(pred)
    
    if pred==0:
        pred="The leaf is diseased cotton leaf"
    elif pred==1:
        pred="The leaf is diseased cotton plant"
    elif pred==2:
        pred="The leaf is fresh cotton leaf"
    else:
        pred="The leaf is fresh cotton plant"
        
    return pred


# In[4]:


# Define the main app
app = Flask(__name__,template_folder='views')


# In[5]:


# Define the end points
@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    # Get the file from the request
    file = request.files['image']
    
    # Save the file in uploads folder
    CURR_DIR = os.path.abspath('')
    file_path = os.path.join(CURR_DIR,'uploads',secure_filename(file.filename))
    file.save(file_path)
    
    # Replace this with actual model
    model=None
    
    # Predict
    prediction = predict_disease(file_path, model)
    
    return prediction


# In[ ]:


# Start the App in DEBUG mode.
if __name__=="__main__":
    app.run(debug=True, use_reloader=False)


# In[ ]:





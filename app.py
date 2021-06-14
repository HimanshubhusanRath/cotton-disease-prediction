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
from fastai.vision.all import *


# In[2]:


if not os.path.exists('uploads'):
    os.makedirs('uploads')


# In[3]:


# Load the model
model = load_learner('model/cotton_disease_densenet-model.h5')


# In[4]:


def predict_disease(image_path, model):
    img = plt.imread(image_path)
    classofimg, idx, probability = model.predict(img)
        
    return classofimg


# In[5]:


# Define the main app
app = Flask(__name__,template_folder='views')


# In[6]:


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
    
    # Predict
    prediction = predict_disease(file_path, model)
    
    return prediction


# In[ ]:


# Start the App in DEBUG mode.
if __name__=="__main__":
    app.run(debug=True, use_reloader=False)


# In[ ]:





# In[ ]:





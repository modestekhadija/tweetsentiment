import numpy as np
from flask import Flask, request, render_template
from preprocessing import *
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentiment',methods=['GET','POST'])
def sentiment():

    tweet = [x for x in request.form.values()]
    output = predict_text(tweet[0])
  
    return render_template('index.html', prediction_text='This is a {} tweet'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
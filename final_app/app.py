# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_dic= ["Benign","Melignant"]



from model_predict  import pred_fn

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Palm Based Person Identification'
    return render_template('index.html', title=title)

# render crop recommendation form page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Palm_detection'

    if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)
            file = request.files.get('file')

            print(file)
        #if not file:
         #   return render_template('disease.html', title=title)
        #try:
            img1 = file.read()
            with open('input.png', 'wb') as f:
                    f.write(img1)
            #print(img)
            #input_image1("input.png")
            from feature_extraction1 import palm_lines

            palm_lines()
            waiting_time = 5
            print(f"Waiting for {waiting_time} seconds...")
            time.sleep(waiting_time)
            prediction =pred_fn("input.png")


            return render_template('disease-result.html', prediction="Person Identified",precaution=prediction,title=title)
        #except:
         #   pass
    return render_template('disease.html', title=title)


# render disease prediction result page


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)

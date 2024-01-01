from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.mlproject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        # try:
            #  reading the inputs given by the user
            year =int (request.form['year'])
            present_price =float (request.form['present_price'])
            kms_driven =int (request.form['kms_driven'])
            fuel_type =int (request.form['fuel_type'])
            seller_type =int (request.form['seller_type'])
            transmission =int (request.form['transmission'])
            owner =int (request.form['owner'])
            
            
            data = [year,present_price,kms_driven,fuel_type,seller_type,transmission,owner]
            # data=np.array(data).reshape(1,-1)
            data = np.array(data).reshape(1, 7)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)
            predict = predict[0]
            return render_template('result.html', prediction = str(predict))

        # except Exception as e:
        #     print('The Exception message is: ',e)
        #     return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)
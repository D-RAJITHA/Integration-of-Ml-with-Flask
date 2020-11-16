from flask import Flask,render_template,request
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("/home/m0162/Desktop/Rajitha/Desktop/Project/Churn prediction/randomforestchurnmodel.pkl")
#model = load_model('E:\\banking analysis\\banking_analysis_model@@@.h5')

@app.route('/')
def home():
    return render_template('churn.html')


@app.route('/predict',methods=['POST'])
def predict():
    print(request.form)

    fname=request.form['fname']
    print(fname)
    Loantype=int(request.form['Loantype'])
    print(Loantype)
   
  
    CreditScore=float(request.form['CreditScore'])
    print(CreditScore)
    
    
    Age=int(request.form['Age'])
    print(Age)
    
    Gender=int(request.form['Gender'])
    print(Gender)
      
    Geography=int(request.form['Geography'])
    print(Geography)
    
    NumOfProducts=int(request.form['NumOfProducts'])
    print(NumOfProducts)
    
    Tenure1=int(request.form['Tenure1'])
    print(Tenure1)
    
    Tenure2=int(request.form['Tenure2'])
    print(Tenure2)
    
    Tenure3=int(request.form['Tenure3'])
    print(Tenure1)
    
    Tenure4=int(request.form['Tenure4'])
    print(Tenure4)
    
    PayableDebt1=float(request.form['PayableDebt1'])
    print(PayableDebt1)
    
    PayableDebt2=float(request.form['PayableDebt2'])
    print(PayableDebt2)
    
    PayableDebt3=float(request.form['PayableDebt3'])
    print(PayableDebt3)
    
    PayableDebt4=float(request.form['PayableDebt4'])
    print(PayableDebt4)
    
    
    
    EstimatedSalary=float(request.form['EstimatedSalary'])
    print(EstimatedSalary)
    
    
    
  
   
    
   
    data = [[CreditScore, Geography, Gender, Age, Tenure1, Tenure2, Tenure3, Tenure4,PayableDebt1, PayableDebt2, PayableDebt3, PayableDebt4, NumOfProducts, EstimatedSalary, Loantype]]
    df = pd.DataFrame(data, columns = ['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure1', 'Tenure2', 'Tenure3', 'Tenure4',
       'PayableDebt1', 'PayableDebt2', 'PayableDebt3', 'PayableDebt4',
       'NumOfProducts', 'EstimatedSalary','Loantype']) 
       
    print(df)      
    
    #y_pred1 = model.predict(df)
    y_pred=model.predict_proba(df)
    print(y_pred)
    output = y_pred[:,0]*100
    a_list = list(output)
    print(a_list[0])
    a= round(a_list[0], 2) 
   
  
    
    #return render_template('index4.html', prediction_text ='The probability of getting churn is:{}'.round(a_list[0], 2))
    return render_template('churn.html', prediction_text='The probability of a customer becoming defaulter is :{}%'.format(a))
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
    
if __name__ == '__main__':
    app.run()
    
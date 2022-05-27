import flask
from flask import Flask, jsonify, request, render_template, request, redirect, url_for
import json
from data_input import data_in
import numpy as np
import pickle

#Mehod to load the model for the app
def load_models():
    file_name = 'models/model_file.p'
    
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
        
    return model

#Method to load select scalers for the app to properly scale the data prior to prediction
def load_scaler_plus(scaler_name, pickle_name):
    file_name = scaler_name #'models/scaler.pkl'
    
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        scaler = data[pickle_name[:-4]] #data['scaler']
    return scaler


#Method to make a prediction using a sample of data
app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict(sample):
    #request_json = request.get_json()
    #x = request_json['input']
    
    #x_in = np.array(x).reshape(1, -1)
    
    model = load_models()
    prediction = model.predict(sample) #[0].tolist()
    #response = json.dumps({'response': prediction})
    
    #return response, 200
    
    return prediction

#Method to render the home page containing the form to submit a data sample
@app.route('/')
def home():
    return render_template('index.html')

#Method to create and format a data sample and call the result page to display the resulting prediction
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        
        #Loading all necessary scalers to make predictions
        km_scaler = load_scaler_plus('models/km_scaler.pkl', 'km_scaler.pkl')
        kmeans = load_scaler_plus('models/kmeans.pkl', 'kmeans.pkl')
        lbl_encoder = load_scaler_plus('models/lbl_encoder.pkl', 'lbl_encoder.pkl')
        lda = load_scaler_plus('models/lda.pkl', 'lda.pkl')
        
        sample_scaler = load_scaler_plus('models/sample_scaler.pkl', 'sample_scaler.pkl')
        
        
        #Take in the submitted form's data and transform appropriately
        initial_data = request.form.to_dict()
        initial_data = list(initial_data.values())
        
        initial_data = list(map(float, initial_data))
        
        #print(initial_data)
        
        initial_data_scaled = km_scaler.transform(np.array(initial_data).reshape(1, -1))
        
        #print(initial_data_scaled)
        
        #Transform and create proper fields for kmeans cluster and LDA component scores
        kmeans_point = int(kmeans.predict(initial_data_scaled))
        
        #print(kmeans_point)
        print(initial_data)
        print(kmeans_point)
        initial_data.append(kmeans_point)
        
        print(initial_data)
        
        
        lda_points = lda.transform(np.array(initial_data).reshape(1, -1)).tolist()
        ld1, ld2 = lda_points[0][0], lda_points[0][1]
        initial_data.append(ld1)
        initial_data.append(ld2)
        
        print(initial_data)
        
        #Scale the final resulting array of data to then make a prediction and display it
        final_data = sample_scaler.transform(np.array(initial_data).reshape(1, -1))
        
        print(final_data)
        
        prediction = int(predict(final_data))
        
        pred_mapper = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
        final_pred = pred_mapper[prediction]
        #print(int(prediction))
        
        return render_template('result.html', prediction = final_pred)
        

if __name__ == '__main__':
    app.run(debug=True)
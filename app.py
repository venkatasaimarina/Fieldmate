from turtle import title
from flask import Flask, config, render_template, request
import pickle
import numpy as np
import joblib
import numpy as np
import requests
from werkzeug.utils import secure_filename

from flask import Flask, redirect, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
app = Flask(__name__, static_url_path='/static')

fertlizer_model = joblib.load("models/fertilizer_prediction_model.pkl")
fertlizer_model_org=joblib.load("models/fertilizer_prediction_organic_model.pkl")
# Load the trained Decision Tree model
model_file_path = "models\crop_recommendation_model.pkl"
with open(model_file_path, 'rb') as model_file:
    crop_recommendation_model = pickle.load(model_file)


model_file_path_yield = "models/yeild_prediction_model.pkl"
with open(model_file_path_yield, 'rb') as model_file_yield:
    rfr_yield = pickle.load(model_file_yield)
disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']
disease_model_path = "models\plant_disease_model (2).pth"
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu'), weights_only=True))
disease_model.eval()
def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# Function to predict the crop based on input data
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = crop_recommendation_model.predict(input_data)
    return prediction[0]


def predict_yield(State_Name, District_Name, Crop_Year, Season, Crop, Area):
    input_data = np.array([[State_Name, District_Name, Crop_Year, Season, Crop, Area]])
    prediction = rfr_yield.predict(input_data)
    return prediction[0]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sample')
def index1():
    return render_template('crop_recommend.html')

@app.route('/fertilizer_result')
def index2():
    return render_template('fertilizer_prediction_result.html')

@app.route('/fertilizer_result_organic')
def index3():
    return render_template('fertilizer_prediction_organic_result.html')

@app.route('/yeild_result')
def index4():
    return render_template('yield_recommendation.html')
@app.route('/image-predict', methods=['GET', 'POST'])
def index5():
    return render_template('leaf_disease.html')
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'FeildMate- Disease Detection'
    prediction = None  # Initialize with a default value

    if request.method == 'POST':
        file = request.files.get('file')
        if file is not None:  # Check if file exists
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
        else:
            # Handle case where no file is selected
            prediction = "No file selected. Please upload an image."


    return render_template("open2.html", prediction=prediction, title=title)




def weather_fetch(city_name):
    try:
        api_key = "bc552d5a6eb478cfe2744823a98ae3f6"
        base_url = "https://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}q={city_name}&appid={api_key}"

        response = requests.get(complete_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()
        if "main" in data:
            main_data = data["main"]
            temperature = round(main_data["temp"] - 273.15, 2)  # Convert temperature to Celsius
            humidity = main_data["humidity"]
            return temperature, humidity
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        try:
            # Extract input data from form
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['pottasium'])
            #temperature = float(request.form['Temperature'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            #humidity = float(request.form['Humidity'])
            city = request.form.get("city")

            # Fetch weather data
            weather_data = weather_fetch(city)
            if weather_data:
                temperature, humidity = weather_data

                # Prepare input data for prediction
                data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                my_prediction = crop_recommendation_model.predict(data)
                final_prediction = my_prediction[0]

                # Render template with prediction result
                return render_template('crop_prediction_result.html', prediction=final_prediction, title=title)
            else:
                return render_template('error.html', message="Error fetching weather data.")
        except KeyError as e:
            return render_template('error.html', message=f"Missing form field: {e}")
        except ValueError as e:
            return render_template('error.html', message=f"Invalid input value: {e}")
        except Exception as e:
            return render_template('error.html', message=f"An unexpected error occurred: {e}")
        
@app.route("/predict-fertilizer")
def kidney():
    return render_template("fertilizer_recommendation.html")

@app.route("/predictkidney",  methods=['GET', 'POST'])
def predictkidney():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        to_predict = np.array(to_predict_list).reshape(1, 8)
        # loaded_model = joblib.load('.pkl')
        data = fertlizer_model.predict(to_predict)
        da=fertlizer_model_org.predict(to_predict)
        return render_template("fertilizer_prediction_result.html", name=data[0],na=da[0])
@app.route('/predict-fertilizer_organic', methods=['POST'])    
@app.route('/predict-yield', methods=['POST'])
def yield_prediction():
    title = 'Yield Prediction'
    if request.method == 'POST':
        State_Name = (request.form['statename'])
        District_Name = (request.form['District_name'])
        Crop_Year = int(request.form['crop_year'])
        Season = int(request.form['season'])
        Crop = int(request.form['crop'])
        Area = float(request.form['Area'])
        State_Name=0
        dt={'ANANTAPUR':0, 'CHITTOOR':1, 'EAST GODAVARI':2, 'GUNTUR':3, 'KADAPA':4,
       'KRISHNA':5, 'KURNOOL':6, 'PRAKASAM':7, 'SPSR NELLORE':8, 'SRIKAKULAM':9,
       'VISAKHAPATANAM':10, 'VIZIANAGARAM':11, 'WEST GODAVARI':12}
        # Use the predict_yield function to get the prediction
        prediction_yield = predict_yield(State_Name, dt[District_Name], Crop_Year, Season, Crop, Area)

        return render_template('yield_prediction_result.html', prediction_yield=prediction_yield, title=title)

if __name__ == "__main__":
    app.run(debug=True)

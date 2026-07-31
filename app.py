import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import model as model_module

app = Flask(__name__)

# ---------------------------------------------------------
# 1. Model Loading Logic
# ---------------------------------------------------------
IGNORED_CLASSES = {'AdamState', 'Adam', 'SGD', 'Optimizer'}

ModelClass = None
for attr_name in dir(model_module):
    attr = getattr(model_module, attr_name)
    if isinstance(attr, type) and attr.__module__ == 'model' and attr_name not in IGNORED_CLASSES:
        ModelClass = attr
        print(f"Detected Neural Network Class: {attr_name}")
        break

MODEL_PATH = "poshannet_model.npz"
model = None

if os.path.exists(MODEL_PATH) and ModelClass is not None:
    try:
        model = ModelClass(input_dim=6, hidden_dim=32, output_dim=3)
    except TypeError:
        model = ModelClass()

    if model is not None:
        try:
            if hasattr(model, 'load_weights'):
                model.load_weights(MODEL_PATH)
            elif hasattr(model, 'load'):
                model.load(MODEL_PATH)
            print("PoshanNet weights loaded successfully!")
        except Exception as e:
            print(f"Error loading weights: {e}")

# ---------------------------------------------------------
# 2. Clinical & WHO Standard Calculations
# ---------------------------------------------------------
def calculate_who_zscore(height_cm, weight_kg, age_months, sex):
    # Standard WHO Median Estimations
    exp_height = 50 + (age_months * 1.5)
    exp_weight = 3.3 + (age_months * 0.5)
    
    haz = (height_cm - exp_height) / 3.5
    waz = (weight_kg - exp_weight) / 1.2
    whz = (weight_kg / (height_cm / 100)**2 - 15) / 1.5
    
    return round(haz, 2), round(waz, 2), round(whz, 2)

def diagnose_status(prediction_class, muac_cm, whz):
    labels = {0: "Normal", 1: "Moderate Acute Malnutrition (MAM)", 2: "Severe Acute Malnutrition (SAM)"}
    status = labels.get(prediction_class, "Normal")
    
    # WHO Clinical Overrides
    if muac_cm > 0 and muac_cm < 11.5:
        status = "Severe Acute Malnutrition (SAM) [Critical MUAC]"
    elif 11.5 <= muac_cm < 12.5 and status == "Normal":
        status = "Moderate Acute Malnutrition (MAM) [Low MUAC]"
        
    return status

# ---------------------------------------------------------
# 3. Routes
# ---------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        muac = float(request.form['muac'])
        head_circ = float(request.form['head_circ'])
        
        haz, waz, whz = calculate_who_zscore(height, weight, age, sex)
        features = np.array([[age, sex, height, weight, muac, head_circ]], dtype=np.float32)
        
        if model is not None:
            if hasattr(model, 'forward'):
                probs = model.forward(features)
            elif hasattr(model, 'predict'):
                probs = model.predict(features)
            else:
                probs = model(features)
                
            pred_class = int(np.argmax(probs, axis=1)[0])
            confidence = round(float(np.max(probs)) * 100, 1)
        else:
            pred_class = 0
            confidence = 0.0
            
        status = diagnose_status(pred_class, muac, whz)
        
        result = {
            'age': age,
            'sex': 'Male' if sex == 1 else 'Female',
            'height': height,
            'weight': weight,
            'muac': muac,
            'head_circ': head_circ,
            'haz': haz,
            'waz': waz,
            'whz': whz,
            'status': status,
            'confidence': confidence
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return f"Error processing input: {str(e)}", 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
import os
import random
from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['IMAGE_FOLDER'] = os.path.join('static', 'images')
app.config['MODEL_PATH'] = 'model.pkl'
app.config['ENCODER_PATH'] = 'label_encoder.pkl'
app.config['DATA_PATH'] = 'real_estate_data.csv'  # Create this file or use sample data

# Sample property images
PROPERTY_IMAGES = ["property1.jpeg", "property2.jpeg", "property3.jpeg", "property4.jpeg"]

def create_sample_data():
    """Create sample data if CSV doesn't exist"""
    data = {
        'area': [1200, 1500, 1800, 2000, 2400, 3000, 3500, 4000, 4500, 5000],
        'bedrooms': [2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        'bathrooms': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        'location': ['Urban', 'Suburban'] * 5,
        'age': [5, 10, 2, 15, 7, 20, 1, 12, 8, 25],
        'price': [2500000, 3200000, 4200000, 3800000, 5500000, 6200000, 7500000, 6800000, 8200000, 7000000]
    }
    return pd.DataFrame(data)

def load_or_create_data():
    try:
        return pd.read_csv(app.config['DATA_PATH'])
    except FileNotFoundError:
        df = create_sample_data()
        df.to_csv(app.config['DATA_PATH'], index=False)
        return df

def train_model():
    df = load_or_create_data()
    
    # Preprocessing
    le = LabelEncoder()
    df['location'] = le.fit_transform(df['location'])
    
    # Save encoder
    with open(app.config['ENCODER_PATH'], 'wb') as f:
        pickle.dump(le, f)
    
    # Train model
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open(app.config['MODEL_PATH'], 'wb') as f:
        pickle.dump(model, f)
    
    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/images/<filename>')
def serve_image(filename):
    try:
        return send_from_directory(app.config['IMAGE_FOLDER'], filename)
    except FileNotFoundError:
        return send_from_directory(app.config['IMAGE_FOLDER'], 'default.jpeg')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'area': float(request.form['area']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'location': request.form['location'],
            'age': int(request.form['age'])
        }
        
        # Load model
        with open(app.config['MODEL_PATH'], 'rb') as f:
            model = pickle.load(f)
        
        # Load encoder
        with open(app.config['ENCODER_PATH'], 'rb') as f:
            le = pickle.load(f)
        
        # Prepare input
        input_df = pd.DataFrame([data])
        input_df['location'] = le.transform([data['location']])
        
        # Predict
        prediction = model.predict(input_df)[0]
        formatted_price = f'₹{prediction:,.2f}'.replace('.00', '')
        
        # Select image
        property_image = random.choice(PROPERTY_IMAGES)
        if not os.path.exists(os.path.join(app.config['IMAGE_FOLDER'], property_image)):
            property_image = 'default.jpeg'
        
        return render_template('result.html',
                            prediction_text=formatted_price,
                            bedrooms=data['bedrooms'],
                            bathrooms=data['bathrooms'],
                            area=data['area'],
                            location=data['location'],
                            age=data['age'],
                            property_image=property_image)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)
    
    # Train model if not exists
    if not os.path.exists(app.config['MODEL_PATH']):
        print("Training model...")
        train_model()
    
    app.run(debug=True)
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model pipeline (ensure the file 'best_model_pipeline.pkl' exists)
with open('best_model_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from the form
    # Expected form fields: number_of_guests, quantity_of_food, type_of_food, event_type,
    # storage_conditions, purchase_history, seasonality, preparation_method, geographical_location, pricing
    input_data = {
        'number_of_guests': float(request.form['number_of_guests']),
        'quantity_of_food': float(request.form['quantity_of_food']),
        'type_of_food': request.form['type_of_food'],
        'event_type': request.form['event_type'],
        'storage_conditions': request.form['storage_conditions'],
        'purchase_history': request.form['purchase_history'],
        'seasonality': request.form['seasonality'],
        'preparation_method': request.form['preparation_method'],
        'geographical_location': request.form['geographical_location'],
        'pricing': request.form['pricing']
    }
    
    # Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Use the loaded model pipeline to make a prediction
    prediction = model.predict(input_df)[0]
    
    # Render the same page with the prediction result
    return render_template('index.html', prediction_text=f'Predicted Food Wastage Amount: {prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})


# Load the trained model and prepare data
data = pd.read_csv("data.csv")  # Update the path to reflect the new structure
features = data.drop(['Total'], axis='columns')
target = data["Total"]
new_features = pd.get_dummies(features, drop_first=True)

# Train the model
model = BaggingRegressor()
model.fit(new_features, target)
@app.route('/')
def index():
    return {"name": "hello"}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user responses from the request
        user_responses = [int(request.json[f'col{i}']) for i in range(10)]

        # Make a prediction using the trained model
        input_data = pd.DataFrame([user_responses], columns=new_features.columns)
        predicted_total = model.predict(input_data)[0]

        print('Predicted Total:', predicted_total)

        return jsonify({'predicted_total': predicted_total})
    except Exception as e:
        print('Error predicting result:', str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

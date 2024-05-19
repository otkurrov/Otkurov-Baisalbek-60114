from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('predict', methods=['POST'])
def predict():
    
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No JSON data provided'}), 500

    
    df = pd.DataFrame(data, index=[0])

    try:
        prediction = model.predict(df)
        return jsonify({'Credit_Score': prediction[0]}), 200  
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

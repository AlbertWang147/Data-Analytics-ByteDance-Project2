import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
f = open(f"model/lr1.pkl", "rb")
model = pickle.load(f)
f.close()

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    features_list = list(request.form.values())
    features = np.array(features_list).reshape(1, -1)
    predict_outcome = model.predict(features)
    return render_template('page.html',
                           prediction_display_area='Predict Outcome：{}'.format(predict_outcome)
                           )
if __name__ == "__main__":
    app.run(port=5700, debug=True)

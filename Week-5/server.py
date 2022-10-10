from flask import Flask
from flask import jsonify
from flask import request
# import pickle
from answers import log_reg_model, dict_vector_model

# input_model = "model1.bin"
# dict_model = "dv.bin"

# with open(input_model, "rb") as f:
#     log_reg_model = pickle.load(f)

# with open(dict_model, "rb") as f:
#     dict_vector_model = pickle.load(f)

def predict(client):
    X = dict_vector_model.transform(client)
    pred_prob = log_reg_model.predict_proba(X)
    return pred_prob

# Flask App
app = Flask('testing')

@app.route('/predict', methods=['POST'])

def predict():
    client = request.get_json()
    x_test = dict_vector_model.transform(client)
    y_pred_proba = log_reg_model.predict_proba(x_test)[:,1]
    result = {
        'card_approved_probability': float(y_pred_proba)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=9696)

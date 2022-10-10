import pickle
import requests

input_model = "model1.bin"
dict_model = "dv.bin"

with open(input_model, "rb") as f:
    log_reg_model = pickle.load(f)

with open(dict_model, "rb") as f:
    dict_vector_model = pickle.load(f)

def predict(values):
    X = dict_vector_model.transform(values)
    pred_prob = log_reg_model.predict_proba(X)
    return pred_prob

data = {
    "reports": 0,
    "share": 0.001694,
    "expenditure": 0.12,
    "owner": "yes"
}


probablity = predict(data)
print(probablity[0][1])


# Server
# url = "http://localhost:9696/predict"

# client = {
#     "reports": 0,
#     "share": 0.245,
#     "expenditure": 3.438,
#     "owner": "yes"
# }

# print(requests.post(url, json=client).json())

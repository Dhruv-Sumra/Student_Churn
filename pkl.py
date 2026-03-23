import pickle

# Load the file
with open('churn_model.pkl', 'rb') as file:
    data = pickle.load(file)

# Check what it contains
print(type(data))
print(data)
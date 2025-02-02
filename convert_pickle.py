import pickle
import gzip
import json

# Load the data from the .pkl.gz file
with gzip.open("mnist.pkl.gz", "rb") as f:
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

# Define a helper function to convert the data into JSON serializable format
def convert_data(data):
    features, labels = data
    return [{"x": features[i].tolist(), "y": int(labels[i])} for i in range(len(features))]

# Convert and save to JSON
with open("training_data.json", "w") as train_json:
    json.dump(convert_data(training_data), train_json)

with open("validation_data.json", "w") as val_json:
    json.dump(convert_data(validation_data), val_json)

with open("test_data.json", "w") as test_json:
    json.dump(convert_data(test_data), test_json)
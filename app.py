from flask import Flask, render_template, request, jsonify

# Your existing code here

# Importing required libraries
import numpy as np

# Function to convert a decimal to binary, it will be required in data generation
def dec_to_bin(n, size):
  num = np.zeros(size)
  for j in range(size):
    num[size - j-1] = n%2
    n = int(n/2)
  return num

# Function to generate data
def data_generation(n):

  # Defining shape of input and output array
  input_array = np.zeros((2**n,n))
  output_array = np.zeros((2**n,(2**(2**n))))

  # Assigning binary values to input and output arrays using decimal to binary converter function
  for i in range(input_array.shape[0]):
    input_array[i][:] = dec_to_bin(i,input_array.shape[1])
  for i in range(output_array.shape[1]):
    output_array[:,i] = dec_to_bin(i,output_array.shape[0])

  # Returning input and output array
  return input_array, output_array


"""## Binary Classifier
Now we will be building our one layer neural network.

### Activation function
![Sigmoid Function](https://th.bing.com/th/id/OIP.HXCBO-Wx5XhuY_OwMl0PhwHaE8?w=229&h=180&c=7&r=0&o=5&dpr=1.5&pid=1.7)
"""

def sigmoid(x):
  return 1/(1+ np.exp(-x))

"""### Loss function
![Alt Text](https://rb.gy/kwz0a)

"""

def binary_cross_entropy(y_true,y_pred):
  return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

"""## Function to calculate number of classification and misclassfiction"""

def Classifier(n, neta=0.3, epochs=50):
    # Generate input data 'x' and corresponding labels 'Y'
    x, Y = data_generation(n)

    # Initialize counters for classified and misclassified data points
    classified = 0
    misclassified = 0

    # Set the learning rate
    learning_rate = neta

    # Loop through each data point
    for i in range(Y.shape[1]):
        y = Y[:, i].reshape(-1, 1)  # Select the label for the current data point

        # Initialize random weights and bias
        weights = np.random.randn(x.shape[1], 1)
        bias = np.random.randn()

        num_epochs = epochs
        loss_ = []  # List to store the loss for each epoch

        # Training loop over epochs
        for epoch in range(num_epochs):
            # Calculate predictions using the current weights and bias
            predictions = sigmoid(np.dot(x, weights) + bias)

            # Calculate binary cross-entropy loss
            loss = binary_cross_entropy(y, predictions)
            loss_.append(loss)

            # Calculate gradients
            dw = np.dot(x.T, predictions - y)
            db = np.sum(predictions - y)

            # Update weights and bias using gradient descent
            weights -= learning_rate * dw
            bias -= learning_rate * db

        # Calculate predictions using the trained weights and bias
        y_predicted = sigmoid(np.dot(x, weights) + bias)

        # Normalize predictions to [0, 1] range
        min_val = np.min(y_predicted)
        max_val = np.max(y_predicted)
        normalized_y_predicted = (y_predicted - min_val) / (max_val - min_val)

        # Apply a threshold to classify normalized predictions into 0 or 1
        for i in range(len(normalized_y_predicted)):
            if normalized_y_predicted[i] <= 0.5 or y_predicted[i] < 0.1:
                normalized_y_predicted[i] = 0
            if normalized_y_predicted[i] > 0.5 or y_predicted[i] > 0.8:
                normalized_y_predicted[i] = 1

        # Compare predicted labels with actual labels
        if np.all(y == normalized_y_predicted):
            classified += 1
        else:
            misclassified += 1

    # Return the counts of classified and misclassified data points
    return classified, misclassified

def count_classification(n):
  results = []
  for i in n:
    C,M = Classifier(i)
    result_dict = {
            "input_count": i,
            "correct_count": C,
            "misclassified_count": M
        }
    results.append(result_dict)
    # print(f"For {i} input gates, {C} functions have been correctly classified, while {M} functions have been misclassified.")
  return results

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/get_results", methods=["POST"])
def get_results():
    input_list = request.form.get("input_list")
    input_numbers = [int(x) for x in input_list.split(",")]
    results = count_classification(input_numbers)
    
    result_strings = []
    for result in results:
        result_string = f"For {result['input_count']} input gates, {result['correct_count']} functions have been correctly classified, while {result['misclassified_count']} functions have been misclassified."
        result_strings.append(result_string)
    
    return jsonify(result_strings)

if __name__ == "__main__":
    app.run(debug=True)

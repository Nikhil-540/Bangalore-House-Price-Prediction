import pandas as pd
from flask import Flask,render_template,request
import sklearn
import pickle
import pandas
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
data = pd.read_csv("Clean_data.csv")
pipe = pickle.load(open("XG.pkl", "rb"))


@app.route("/")
def Hello():
    locations = sorted(data["location"].unique())
    return render_template("web.html",locations=locations)

@app.route("/predict",methods =["POST"])
def predict():
    location = request.form.get("location")
    bhk = request.form.get("bhk")
    bath= request.form.get("bath")
    sqft =request.form.get("total_sqft")

    # data = {
    #     'Location': [location],  # Replace 'Location' with the actual column name for location
    #     'size': [bhk],  # Replace 'BHK' with the actual column name for BHK
    #     'bath': [bath],  # Ensure the column name matches the expected column name
    #     'total_sqft': [sqft]  # Ensure the column name matches the expected column name
    # }

    # Create a DataFrame using the form data
    input_df = pd.DataFrame([[location,bhk,bath,sqft]],columns=["location","size","total_sqft","bath"])
    prediction = pipe.predict(input_df)
    return render_template('result.html', prediction=prediction)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get input data from the form
#     location = request.form.get("location")
#     bhk = (request.form.get("bhk"))
#     bath= (request.form.get("bath"))
#     sqft =(request.form.get("total_sqft"))
#
#     print(location,bhk,bath,sqft)
    # data = {
    #     'Input1': [location],
    #     'Input2': [bhk],
    #     'Input3': [bath],
    #     'Input4': [sqft]
    # }
    # input_df = [[location, bhk, bath, sqft]]
    # # df =pd.DataFrame(location,bhk,bath,sqft)
    # # Preprocess data if needed
    #
    # # Pass the data to your ML model for prediction
    # prediction = pipe.predict(input_df)

    # Return prediction result to the user
    # return render_template('result.html', prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)
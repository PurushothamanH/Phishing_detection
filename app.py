from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        try:
            # Collect data from form
            data = CustomData(
                NumDots = int(request.form.get('NumDots',0)),
                PathLevel = int(request.form.get('PathLevel', 0)),  # Convert to integer
                UrlLength = int(request.form.get('UrlLength',0)),
                NumDash = int(request.form.get('NumDash',0)),
                NumNumericChars=int(request.form.get('NumNumericChars', 0)),  # Convert to integer
                RandomString = int(request.form.get('RandomString',0)),
                DomainInPaths = int(request.form.get('DomainInPaths',0)),
                PathLength = int(request.form.get('PathLength',0)),
                PctExtHyperlinks = int(request.form.get('PctExtHyperlinks',0)),
                InsecureForms = int(request.form.get('InsecureForms',0)),
                RelativeFormAction = int(request.form.get('RelativeFormAction',0)),
                PctNullSelfRedirectHyperlinks = int(request.form.get('PctNullSelfRedirectHyperlinks',0)),
                FrequentDomainNameMismatch = int(request.form.get('FrequentDomainNameMismatch',0)),
                SubmitInfoToEmail= int(request.form.get('SubmitInfoToEmail',0)),
                IframeOrFrame = int(request.form.get('IframeOrFrame',0)),
                UrlLengthRT = int(request.form.get('UrlLengthRT',0)),
                PctExtResourceUrlsRT = int(request.form.get('PctExtResourceUrlsRT',0)),
                ExtMetaScriptLinkRT = int(request.form.get('ExtMetaScriptLinkRT',0)),  # Convert to int
                PctExtNullSelfRedirectHyperlinksRT = int(request.form.get('PctExtNullSelfRedirectHyperlinksRT', 0))  # Convert to int
            )

            # Convert data to DataFrame
            pred_df = data.get_data_as_data_frame()
            # print(pred_df)

            # Load prediction pipeline and get results
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            if results[0] > 0.5:
                result = "Phishing happened"
            else:
                result = "No Phishing happened"

            # Render results in home.html
            return render_template('home.html', results=result)
            

        except Exception as e:
            print(f"Error occurred: {e}")
            return render_template('home.html', results="Error during prediction. Please check input values.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

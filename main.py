from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

app = FastAPI()

# Load the dataset
data_set = pd.read_csv('diseases.csv')

# Load the trained model
loaded_model = joblib.load('model.pkl')
# Load the vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Define Pydantic BaseModel for request body
class SymptomInput(BaseModel):
    symptoms: str

# Define Pydantic BaseModel for response body
class DiseasePrediction(BaseModel):
    disease: str
    precaution: str

# Endpoint for predicting diseases based on symptoms
@app.post("/predict_disease/", response_model=DiseasePrediction)
async def predict_disease(symptoms_input: SymptomInput):
    try:
        # Preprocess the input symptoms
        input_text = [symptoms_input.symptoms]
        input_vectorized = vectorizer.transform(input_text)

        # Make predictions using the trained model
        predicted_disease = loaded_model.predict(input_vectorized)

        # Get the index of the predicted disease
        disease_index = data_set[data_set['Disease'] == predicted_disease[0]].index[0]

        # Get the corresponding precaution for the predicted disease
        precaution = data_set.loc[disease_index, 'Precaution']

        return {"disease": predicted_disease[0], "precaution": precaution}
    except Exception as e:
        return {"error": str(e)}

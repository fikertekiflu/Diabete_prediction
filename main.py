from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# CORS middleware for handling cross-origin requests
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model to define the input structure
class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load the saved diabetes prediction model
try:
    with open('diabetes_model.sav', 'rb') as f:
        diabetes_model = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

@app.post('/smart-symptomChecker/diabetes')
async def diabetes_predict(input_parameters: ModelInput):
    try:
        # Convert input data to dictionary and list for prediction
        input_data = input_parameters.dict()
        input_list = [
            input_data['Pregnancies'],
            input_data['Glucose'],
            input_data['BloodPressure'],
            input_data['SkinThickness'],
            input_data['Insulin'],
            input_data['BMI'],
            input_data['DiabetesPedigreeFunction'],
            input_data['Age']
        ]

        # Perform prediction
        prediction = diabetes_model.predict([input_list])

        # Generate appropriate response
        if prediction[0] == 0:
            return JSONResponse(content={"result": "The person is not diabetic"}, status_code=200)
        else:
            return JSONResponse(content={"result": "The person is diabetic"}, status_code=200)

    except Exception as e:
        # Error handling in case of prediction failure
        return JSONResponse(content={"error": str(e)}, status_code=500)

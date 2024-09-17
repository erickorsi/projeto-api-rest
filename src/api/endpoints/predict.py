'''
This file contains the /predict route.
'''
import os
import glob
import pickle
import numpy as np
from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from src.api.router_models.input_models.wine_features import Features

predict_router = APIRouter()

@predict_router.post(
    "/", response_model=dict, status_code=status.HTTP_200_OK
)
# Add cache
async def post(
    request: Request,
    features: Features
):
    feature_values = np.array([
        features.feature_1,
        features.feature_2,
        features.feature_3,
        features.feature_4,
        features.feature_5,
        features.feature_6,
        features.feature_7,
        features.feature_8,
        features.feature_9,
        features.feature_10,
        features.feature_11,
        features.feature_12,
        features.feature_13
    ])

    # Load latest model
    # If the model is stored in a database, this process would
    # be done with differently
    pattern = os.path.join("notebook/models", '*.pkl')
    pickle_files = glob.glob(pattern)
    if not pickle_files:
        raise FileNotFoundError("No pickle files were found in the directory.")

    try:
        model_path = max(pickle_files, key=os.path.getmtime)
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
    except Exception as e:
        return JSONResponse({"error": str(e)})

    # Run prediction
    try:
        prediction = model.predict(feature_values)[0]
        return JSONResponse({"prediction": int(prediction)})
    except Exception as e:
        return JSONResponse({"error": str(e)})

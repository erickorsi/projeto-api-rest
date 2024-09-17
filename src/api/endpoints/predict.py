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
        features.alcohol,
        features.malic_acid,
        features.ash,
        features.alcalinity_of_ash,
        features.magnesium,
        features.total_phenols,
        features.flavanoids,
        features.nonflavanoid_phenols,
        features.proanthocyans,
        features.color_intensity,
        features.hue,
        features.od280_od315_of_diluted_wines,
        features.proline
    ]).reshape(1, -1)

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

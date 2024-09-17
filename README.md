# projeto-api-rest
Module for XPInc MLOps project.

This project is an API that receives a JSON of features and returns a prediction using a machine learning model.

# Local Testing
To run the test locally, create a virtual environment (.venv in VSCode for example) with all requirements in the requirements.txt file.

Enter the virtual environment and type in the terminal:
```
uvicorn main:app --reload
```

Using Postgres, import this `curl`, changing the values of each feature according to the desired test:
```
curl -X POST "http://127.0.0.1:8000/predict/" \
-H "Content-Type: application/json" \
-d '{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyans": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.4,
  "proline": 1050
}'
```

# Deploy to Docker
To deploy using Docker, build a docker image by entering the project repository and running:
```
docker build -t fastapi-app .
```

Then run the image as container with:
```
docker run -d --name fastapi-container -p 8000:8000 fastapi-app
```

You can test the API running in docker through Postman, by importing this `curl` and changing the values of features according to the desired test:
```
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "alcohol": 13.0,
  "malic_acid": 1.7,
  "ash": 2.4,
  "alcalinity_of_ash": 15.6,
  "magnesium": 127.0,
  "total_phenols": 2.8,
  "flavanoids": 3.0,
  "nonflavanoid_phenols": 0.3,
  "proanthocyans": 2.6,
  "color_intensity": 4.6,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.1,
  "proline": 1020.0
}'
```

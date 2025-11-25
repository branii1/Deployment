# Netflix Classifier Deployment

A machine learning API service for classifying Netflix content types using a Random Forest model. This project is containerized with Docker and ready for deployment.

## Project Structure

```
netflix_classifier_deployment/
├── Dockerfile              # Docker configuration for containerization
├── predict.py              # Flask API application
├── requirements.txt        # Python dependencies
├── netflix_type_rf_model.pkl  # Trained Random Forest model
└── README.md              # This file
```

## Features

- RESTful API with Flask
- Production-ready deployment with Gunicorn
- Docker containerization
- Health check endpoint
- Prediction endpoint for Netflix content classification

## Prerequisites

- Docker installed on your system
  - [Install Docker Desktop](https://www.docker.com/products/docker-desktop) (Windows/Mac)
  - [Install Docker Engine](https://docs.docker.com/engine/install/) (Linux)
- Docker Hub account (for cloud deployment) or access to a container registry

## Local Deployment

### 1. Build the Docker Image

```bash
docker build -t netflix-classifier:latest .
```

### 2. Run the Container

```bash
docker run -d -p 8080:8080 --name netflix-classifier netflix-classifier:latest
```

The API will be available at `http://localhost:8080`

### 3. Test the API

**Health Check:**
```bash
curl http://localhost:8080/
```

**Get Expected Features:**
```bash
curl http://localhost:8080/features
```

**Make a Prediction:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": "value1", "feature2": "value2", ...}'
```

**Note:** Use the `/features` endpoint to get the exact feature names your model expects. Replace `feature1`, `feature2`, etc. with the actual feature names and values.

### 4. Stop the Container

```bash
docker stop netflix-classifier
docker rm netflix-classifier
```

## Cloud Deployment Options

### Option 1: AWS (Amazon Web Services)

#### Using AWS Elastic Beanstalk

1. **Install EB CLI:**
   ```bash
   pip install awsebcli
   ```

2. **Initialize EB:**
   ```bash
   eb init -p docker netflix-classifier
   ```

3. **Create and deploy:**
   ```bash
   eb create netflix-classifier-env
   eb deploy
   ```

#### Using AWS ECS (Elastic Container Service)

1. **Push image to ECR:**
   ```bash
   aws ecr create-repository --repository-name netflix-classifier
   docker tag netflix-classifier:latest <account-id>.dkr.ecr.<region>.amazonaws.com/netflix-classifier:latest
   aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/netflix-classifier:latest
   ```

2. **Create ECS task definition and service** (via AWS Console or CLI)

#### Using AWS EC2

1. **Launch EC2 instance** (Ubuntu recommended)
2. **Install Docker:**
   ```bash
   sudo apt-get update
   sudo apt-get install docker.io -y
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

3. **Transfer files and build:**
   ```bash
   scp -r . user@ec2-instance:/home/user/netflix-classifier
   ssh user@ec2-instance
   cd netflix-classifier
   docker build -t netflix-classifier .
   docker run -d -p 80:8080 netflix-classifier
   ```

### Option 2: Google Cloud Platform (GCP)

#### Using Cloud Run

1. **Install gcloud CLI** and authenticate
2. **Build and push to Container Registry:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/netflix-classifier
   ```

3. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy netflix-classifier \
     --image gcr.io/PROJECT-ID/netflix-classifier \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8080
   ```

### Option 3: Microsoft Azure

#### Using Azure Container Instances

1. **Install Azure CLI** and login
2. **Create resource group:**
   ```bash
   az group create --name netflix-classifier-rg --location eastus
   ```

3. **Create container registry:**
   ```bash
   az acr create --resource-group netflix-classifier-rg --name netflixclassifier --sku Basic
   ```

4. **Build and push:**
   ```bash
   az acr build --registry netflixclassifier --image netflix-classifier:latest .
   ```

5. **Deploy container instance:**
   ```bash
   az container create \
     --resource-group netflix-classifier-rg \
     --name netflix-classifier \
     --image netflixclassifier.azurecr.io/netflix-classifier:latest \
     --registry-login-server netflixclassifier.azurecr.io \
     --ip-address Public \
     --ports 8080
   ```

### Option 4: Heroku

1. **Install Heroku CLI**
2. **Login and create app:**
   ```bash
   heroku login
   heroku create netflix-classifier-app
   ```

3. **Deploy:**
   ```bash
   heroku container:push web
   heroku container:release web
   ```

4. **Open app:**
   ```bash
   heroku open
   ```

### Option 5: Railway

1. **Sign up at [Railway.app](https://railway.app)**
2. **Connect your GitHub repository** (push this project to GitHub first)
3. **Create new project** and select "Deploy from GitHub repo"
4. **Railway will automatically detect Dockerfile and deploy**

### Option 6: Render

1. **Sign up at [Render.com](https://render.com)**
2. **Create new Web Service**
3. **Connect your GitHub repository**
4. **Configure:**
   - **Build Command:** `docker build -t netflix-classifier .`
   - **Start Command:** `docker run -p $PORT:8080 netflix-classifier`
   - **Environment:** Set `PORT` variable (Render provides this automatically)

## API Endpoints

### GET `/`
Health check endpoint with available endpoints information.

**Response:**
```json
{
  "message": "Netflix Classifier API is running",
  "endpoints": {
    "/": "Health check",
    "/predict": "POST - Make a prediction",
    "/features": "GET - Get expected feature names"
  }
}
```

### GET `/features`
Get the list of expected feature names for the model.

**Response:**
```json
{
  "expected_features": ["feature1", "feature2", "feature3", ...],
  "count": 10
}
```

**Note:** If feature names are not available from the model, the API will return a helpful message.

### POST `/predict`
Make a prediction with comprehensive input validation.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "feature1": "value1",
  "feature2": "value2",
  ...
}
```

**Success Response:**
```json
{
  "input": {
    "feature1": "value1",
    "feature2": "value2"
  },
  "prediction": "predicted_value",
  "prediction_probabilities": {
    "class1": 0.85,
    "class2": 0.15
  }
}
```

**Note:** `prediction_probabilities` is included if the model supports probability predictions.

**Error Responses:**

Missing JSON:
```json
{
  "error": "Content-Type must be application/json"
}
```

Missing required features:
```json
{
  "error": "Missing required features: feature1, feature2"
}
```

Invalid data:
```json
{
  "error": "Feature 'feature1' cannot be None"
}
```

Validation errors:
```json
{
  "error": "Request body is missing or invalid JSON"
}
```

Prediction errors:
```json
{
  "error": "Prediction failed: [error details]",
  "hint": "Please verify that all feature values are in the correct format"
}
```

## Environment Variables

Currently, no environment variables are required. The model path is hardcoded. For production, consider:

- `MODEL_PATH`: Path to the model file
- `PORT`: Port number (default: 8080)
- `WORKERS`: Number of Gunicorn workers (default: 4)

## Troubleshooting

### Container won't start
- Check if port 8080 is already in use: `netstat -ano | findstr :8080` (Windows) or `lsof -i :8080` (Mac/Linux)
- Check Docker logs: `docker logs netflix-classifier`

### Model loading errors
- Ensure `netflix_type_rf_model.pkl` is in the same directory as `predict.py`
- Verify the model file is not corrupted

### Prediction errors
- Verify the input JSON matches the expected feature names and types
- Check the model was trained with the same feature set

## Development

### Running Locally (without Docker)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python predict.py
   ```

3. **Test:**
   ```bash
   curl http://localhost:8080/
   ```

## Input Validation

The API includes comprehensive input validation:

- ✅ **Content-Type validation:** Ensures requests are JSON
- ✅ **Required features check:** Validates all expected features are provided
- ✅ **Data type validation:** Checks for None values and invalid types
- ✅ **Feature discovery:** Automatically extracts expected features from the model
- ✅ **Helpful error messages:** Provides clear feedback on validation failures
- ✅ **Probability support:** Returns prediction probabilities when available

Use the `/features` endpoint to discover what features your model expects before making predictions.

## Improvements for Production

Consider implementing these enhancements:

1. ✅ **Input Validation:** ~~Add schema validation for prediction requests~~ (Implemented)
2. **Logging:** Implement structured logging (e.g., using Python's `logging` module)
3. **Monitoring:** Add health checks and metrics endpoints
4. **Security:** Add authentication/authorization if needed
5. **Rate Limiting:** Implement rate limiting to prevent abuse
6. **CORS:** Configure CORS if serving web clients
7. **Model Versioning:** Support multiple model versions
8. **Caching:** Cache predictions for identical inputs

## License

This project is created for educational purposes as part of a class assignment. 

**Educational Use Only**

This software and associated files are provided for academic and educational purposes. The code, model, and documentation are intended for learning and demonstration purposes only.

**Usage:**
- You may use this code for learning and educational purposes
- You may modify and experiment with the code for your own learning
- Please do not use this for commercial purposes without permission
- If you use this code in your own projects, please provide appropriate attribution

**Disclaimer:**
This project is provided "as is" without warranty of any kind. The author is not responsible for any issues or damages that may arise from the use of this software.

## Author

Bright Siaw


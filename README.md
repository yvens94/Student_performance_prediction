# Math Student Performance Prediction - End-to-End Machine Learning Project

## Objective
- Develop a machine learning project with CI/CD pipelines to predict math grades using the student performance dataset from Kaggle.

## Workflow in Jupyter Notebook
- **Data Exploration**
- **Data Cleaning**: 
- **Feature Engineering**
- **Model Training**
- **Model Evaluation**
- **Hyperparameter Tuning**

## Project Modules
- **data_ingestion.py**: Read data from the database.
- **data_transformation.py**: Perform transformation and feature engineering using a column transformer. Artifact: `preprocessor.pkl`.
- **model_trainer.py**: Train, evaluate, and select the best-performing model. Save both the model and a pickle file.

## Prediction Pipeline
- **predict_pipeline**: Take new user-input data, transform it, and make predictions using the preprocessor from the MVP model.

## Additional Components
- **utils.py**: General utility functions.
- **setup.py**: Project setup and dependencies.
- **logger**: Logging functionality.
- **exception**: Custom exception handling.
- **.gitignore**: Specification of files and directories to be ignored by version control.
- **GitHub Workflow**: Automated workflows for CI/CD.

## Application Deployment
- **app.py**: Flask application containerized with Docker.
- **AWS Deployment**:
  - Deployed on AWS Elastic Beanstalk and CodePipeline.
  - Docker image stored on AWS ECR (Elastic Container Registry).
  - Utilizes GitHub Actions for seamless integration.
  
## Streamlit Deployment
- **mathml.py**: Deployed on Streamlit for interactive visualization.



## ---------------------------------notes------------------------------------------------------------

## AWS Deployment Steps with Docker
- **Docker Build**: Ensured Docker image is successfully built.
- **GitHub Workflow**: Integrated with GitHub Actions.
- **IAM User in AWS**: Set up an IAM user with necessary permissions.
- **Docker Setup in EC2**: Executed commands for Docker installation on EC2.
  - *Optional Steps*:
    - Update and upgrade EC2.
  - *Required Steps*:
    - Install Docker.
    - Add the user to the docker group.
    - Configure EC2 as a self-hosted runner.

## GitHub Secrets (Setup in GitHub Repository)
- `AWS_ACCESS_KEY_ID`: AWS access key for authentication.
- `AWS_SECRET_ACCESS_KEY`: AWS secret access key for authentication.
- `AWS_REGION`: AWS region for deployment (e.g., us-east-1).
- `AWS_ECR_LOGIN_URI`: AWS ECR login URI.
- `ECR_REPOSITORY_NAME`: Name of the ECR repository (e.g., mathml-app).

These steps ensure a smooth deployment of the machine learning project on AWS with Docker, integrating CI/CD pipelines for automated workflows.

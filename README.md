# Dog Breed Image Classification using AWS SageMaker 
This project demonstrates how to use AWS SageMaker for image classification by fine-tuning a pretrained ResNet‑50 model using best ML engineering practices. The project leverages SageMaker’s built-in features such as hyperparameter tuning, profiling, and debugging, and it is designed to work with the provided dog breed classification dataset or any other dataset of your choice.

## Project Set Up and Installation
### AWS Setup and SageMaker Studio:

#### Log in to AWS using the course gateway.
#### Open SageMaker Studio.

### Files and Dependencies:
1. train_and_deploy.ipynb - This notebook contains all the required code and the steps performed in this project and their outputs, starting to unzipping the files to test the results on sample images.
2. hpo.py - This script file contains code that will be used by the hyperparameter tuning jobs to train and test/validate the models with different hyperparameters to find the best hyperparameter
3. train_model.py - This script file contains the code that will be used by the training job to train and test/validate the model with the best hyperparameters that we got from hyperparameter tuning
4. inference_endpoint.py - This script contains code that is used by the deployed endpoint to perform some preprocessing (transformations) , serialization- deserialization and predictions/inferences and post-processing using the saved model from the training job.

#### - Download the starter files which include the training scripts (hpo.py, train_model.py), the inference script (inference_endpoint.py), and the Jupyter Notebook (train_and_deploy.ipynb).
#### - Ensure the required Python libraries are installed (e.g., torch, torchvision, smdebug, etc.).

## Dataset Preparation:
### Download or prepare your dataset. The provided dataset is the dog breed classification dataset.
### Upload the dataset to an S3 bucket via the AWS Gateway, ensuring SageMaker can access it.

## Dataset
The default dataset for this project is the dog breed classification dataset. However, the code and project are designed to be dataset-independent so you may use any dataset that suits your needs.

### Access Instructions:
Upload the data (organized in train and test directories) to an S3 bucket, ensuring that the IAM role linked to SageMaker Studio has the necessary permissions.

## Hyperparameter Tuning
### Model and Hyperparameter Overview

#### Model Choice:
The project uses a pretrained ResNet‑50 model from torchvision with the final fully connected layers replaced by a custom module. Only the new layers are fine-tuned, keeping the rest of the network frozen for efficiency.

#### Parameters Tuned:
- The hyperparameter tuning setup involves at least the following parameters:
- Learning Rate (lr): Controls the step size in the optimizer.
- Weight Decay: Regularization parameter to control overfitting.
(Optional: Additional parameters such as eps and batch size have also been included for further optimization.)

### Tuning Details
#### Logging:
The scripts log key metrics (loss and accuracy) and hyperparameter values during training. These logs can be viewed in AWS CloudWatch.

### Visualization:
Screenshots from the SageMaker hyperparameter tuning jobs (showing at least two separate training runs) are included in the submission.

### Best Hyperparameters:
The Notebook extracts the best performing hyperparameters based on the tuning job results.

## Debugging and Profiling
### Debugging Process
#### Model Debugging:
In the train_model.py script, the AWS SageMaker Debugger is used with smdebug to capture training metrics. By registering both the model and loss, the debugger ensures that any anomalies or errors during training are captured.

### Profiling:
SageMaker Profiling is integrated into the training process to generate detailed performance reports. The outputs include profiling HTML/PDF files which are submitted along with the project.

### Results
Provide screenshots or summaries of the profiler and debugging outputs.

Discuss insights such as training convergence, potential bottlenecks, or unusual metric variations that were identified during the profiling process.

Ensure the profiler report is attached in your submission package.

## Model Deployment
### Deployment Overview
#### Endpoint Deployment:
The inference script (inference_endpoint.py) is used to deploy the trained model as a SageMaker endpoint. The model is loaded once, configured to run in evaluation mode (with optional half-precision on GPU), and compiled (if supported).

### Querying the Endpoint:
After deployment, you can query the endpoint with sample inputs. An example using curl or a Python request can be provided in the submission. For instance, in Python you might use:
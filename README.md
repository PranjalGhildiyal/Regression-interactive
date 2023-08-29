# Regression interactive
Interactive Model Dashboard.

Welcome to my Interactive Model Dashboard project! This powerful tool empowers you to seamlessly explore and apply a diverse array of machine learning models to your dataset. With the added flexibility to dynamically change your target variable on the fly, you're in complete control of your experiments. The dashboard effortlessly opens in your web browser, providing a user-friendly and interactive experience.

One of the standout features of this project is its adaptability. Not only can you choose from a pre-existing selection of machine learning models, but you also have the flexibility to easily import and use any model of your choice. This allows you to tailor your experiments to your unique needs, making the dashboard an ideal platform for both novices and experts in the field.

In addition, the project integrates seamlessly with MLflow, enabling you to conveniently log and manage your model experiments. By leveraging MLflow, you gain the power to track and compare various model configurations effortlessly, making informed decisions about your machine learning strategies.

Whether you're a seasoned data scientist or just beginning your journey, the Interactive Model Dashboard provides an intuitive interface that ensures your machine learning experiments are both enlightening and efficient. I am excited to have you on board as you unlock the potential of this remarkable tool.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Changing Target Variable](#changing-target-variable)
- [MLflow Integration](#mlflow-integration)
- [Adding new models](#adding-new-models)
- [Accessing MLflow UI](#accessing-mlflow-ui)
- [License](#license)

## Introduction

The Interactive Model Dashboard is a powerful tool designed to streamline your machine learning experimentation process. It allows you to import your data, choose from a variety of machine learning models, and apply them to your dataset. Furthermore, you can dynamically change the target variable to observe how different models perform against different objectives.

## Features

- Interactive dashboard to explore and apply machine learning models.
- Dynamic target variable selection for on-the-fly experimentation.
- Integration with MLflow for model tracking and comparison.
- Easy-to-use interface for effortless navigation and model application.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following prerequisites:

- Python >= 3.6 (preferred: 3.9)
- Git

### Installation

1. Clone this repository to your local machine using:

   ```bash
   git clone git@github.com:PranjalGhildiyal/Regression-interactive.git
   ```
2. Navigate to the project directory:
   ```bash
   cd path/to/project/directory/Regression-interactive
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To launch the Interactive Model Dashboard, follow these steps:
1. Open [test.ipynb](example.py)
2. Import the data to experiment on.
3. Use the dashboard to make different models
4. Track models with mlflow by triggering the mlflow ui through the MLflow button on the sidebar.

## Changing Target Variable
The dashboard allows you to change the target variable for your machine learning models interactively. This feature is especially useful for observing how different models perform against different objectives.

To change the target variable, choose the desired target variable from the available options.

## MLflow Integration
All experiments conducted within the Interactive Model Dashboard are automatically logged using MLflow. This integration enables you to:

1. Keep track of different model configurations and their performance.
2. Compare results across different experiments.
3. Visualize metrics and artifacts using the MLflow UI.

## Adding new models
Not finding the existing imported models useful? We got you 😉
1. Go to [model_types.py](config/model_types.py)
2. Import the model you want to use
   
   ![Import Image](https://github.com/PranjalGhildiyal/Regression-interactive/blob/main/Attachments/model_inclusion.png)
   
   (make sure to install any packages required)
3. Include model.
   
   ![Model_Inclusion](https://github.com/PranjalGhildiyal/Regression-interactive/blob/main/Attachments/import_model.png)

#### VOILA! Your new model must be added to the model list in the next run!

## License
This project is licensed under the [MIT](https://opensource.org/license/mit/) License.

## 
I hope you find the Interactive Model Dashboard helpful for your machine learning experiments. If you have any questions or feedback, please don't hesitate to reach out.

Happy experimenting!

Pranjal Ghildiyal.

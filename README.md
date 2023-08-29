# Interactive Model Dashboard with Dynamic Target Variable Selection

Welcome to the **Interactive Model Dashboard** project! This tool allows you to seamlessly explore and apply various machine learning models to your dataset, all while providing the flexibility to change your target variable on the fly. The dashboard opens in your web browser, enabling a user-friendly and interactive experience. Additionally, the project leverages MLflow for model logging and management, giving you the ability to track and compare your model experiments effortlessly.

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
- [Contributing](#contributing)
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

- Python (== 3.9)
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
   
   ![Import Image](https://github.com/PranjalGhildiyal/Regression-interactive/blob/main/Attachments/import_model.png)
   
   (make sure to install any packages required)
3. Include model.
   
   ![Model_Inclusion](https://github.com/PranjalGhildiyal/Regression-interactive/blob/main/Attachments/model_inclusion.png)

### VOILA! Your new model must be added to the model list in the next run!


I hope you find the Interactive Model Dashboard helpful for your machine learning experiments. If you have any questions or feedback, please don't hesitate to reach out.

Happy experimenting!

Pranjal Ghildiyal.

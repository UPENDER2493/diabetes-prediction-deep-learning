# Diabetes Prediction Using Deep Learning

## Week 5 — Deep Learning Application in Data Science

A complete deep learning classification project that applies a PyTorch-based Multilayer Perceptron (MLP) to predict diabetes status using health and lifestyle indicators from the CDC Diabetes Health Indicators dataset.

---

## 📌 Project Overview

This project demonstrates the complete deep learning workflow for a binary classification problem:

**Data Collection → Exploratory Data Analysis → Data Preprocessing → Neural Network Design → Model Training → Evaluation → Threshold Analysis → Critical Analysis**

The objective is to build and evaluate a neural network capable of learning patterns associated with diabetes from demographic, health, lifestyle, and medical-access indicators.

> **Important:** This project is an educational machine-learning exercise and is not intended for medical diagnosis or clinical decision-making.

---

## 🎯 Problem Statement

Diabetes is a major public-health concern, and identifying patterns associated with diabetes status can be useful for data-science research and preventive-health analytics.

The goal of this project is to develop a binary classification neural network that predicts whether an individual belongs to the diabetes-positive class based on 21 health-related and demographic features.

### Target Variable

`Diabetes_binary`

- `0` → No diabetes
- `1` → Diabetes

---

## 📊 Dataset

### Dataset Name

**CDC Diabetes Health Indicators**

### Source

UCI Machine Learning Repository

Dataset ID: **891**

The dataset is publicly available through the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

### Dataset Size

Original dataset:

- Rows: **253,680**
- Columns: **22**
- Input features: **21**
- Target variable: **1**

### Data Quality

- Missing values: **0**
- Duplicate rows: **24,206**
- Duplicate percentage: **9.54%**

Exact duplicate rows were removed before model development.

After duplicate removal:

- Rows: **229,474**
- Features: **21**

### Target Distribution Before Duplicate Removal

| Class | Count | Percentage |
|---|---:|---:|
| No Diabetes (0) | 218,334 | 86.07% |
| Diabetes (1) | 35,346 | 13.93% |

The dataset is therefore imbalanced toward the non-diabetes class.

---

## 🧩 Input Features

The model uses the following 21 features:

```text
HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
GenHlth
MentHlth
PhysHlth
DiffWalk
Sex
Age
Education
Income
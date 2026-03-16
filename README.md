---
title: AcmeTel Churn API
emoji: 📡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AcmeTel Churn Prediction API

Production-ready REST API for customer churn prediction using XGBoost pipeline.

## Endpoints

- `GET /health` — Health check
- `POST /predict` — Predict single customer
- `POST /predict_batch` — Predict batch customers
- `GET /docs` — Swagger UI

## Model

- Algorithm: XGBoost
- ROC AUC: ~0.86
- Features: 19

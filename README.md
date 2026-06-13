# Food101 Rust Deployment

A web application that classifies food images using a fine-tuned MobileNetV3 model trained on the Food101 dataset.

The model was exported to ONNX and deployed using a Rust backend (Axum + ONNX Runtime) and a React frontend.

## Model Performance

| Metric         | Score  |
| -------------- | ------ |
| Top-1 Accuracy | 76.52% |
| Top-5 Accuracy | 93.76% |

Dataset: Food101
Architecture: MobileNetV3 Large

## System Architecture

```text
User
↓
React Frontend (Vite)
↓
Rust Backend (Axum)
↓
ONNX Runtime
↓
MobileNetV3
↓
Food101 Predictions
```

## Features

- Upload food images
- Real-time inference using ONNX Runtime
- Top-5 prediction display
- Confidence scores
- Responsive React UI
- Rust backend API

## Example Predictions

| Image               | Prediction          | Confidence |
| ------------------- | ------------------- | ---------- |
| Pizza               | Pizza               | 99.9%      |
| Paella              | Paella              | ~100%      |
| Gyoza               | Gyoza               | 90.9%      |
| Spaghetti Carbonara | Spaghetti Carbonara | 87.8%      |

## Deployment

## Out-of-Distribution Examples

The model was trained on Food101 and therefore cannot recognize foods outside the dataset.

Examples:

| Actual Food   | Prediction        |
| ------------- | ----------------- |
| Chole Bhature | Breakfast Burrito |
| Dosa          | Hot Dog           |
| Vada Pav      | Beef Tartare      |

These examples demonstrate the limitations of closed-set image classification systems.

## Local Development

### Backend

```bash
cd backend
cargo run
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Screenshots

# Food101 Rust Deployment

A web deployment of a fine-tuned MobileNetV3 Food101 image classifier.

## Stack

- Frontend: React + Vite
- Backend: Rust + Axum
- Inference: ONNX Runtime
- Model: MobileNetV3 fine-tuned on Food101

## Project Structure

```text
backend/   Rust API
frontend/  React web app
model/     ONNX model and class labels
```

# Local Development

## Backend:

cd backend
cargo run

## Frontend:

cd frontend
npm install
npm run dev

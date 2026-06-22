mod image;
mod model;

use crate::image::preprocess::{image_to_tensor, load_image, resize_and_crop};
use crate::model::inference::Prediction;

use axum::extract::DefaultBodyLimit;
use axum::{
    Json, Router,
    extract::{Multipart, State},
    http::StatusCode,
    routing::{get, post},
};
use model::inference::FoodClassifier;
use serde::Serialize;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

#[derive(Clone)]
struct AppState {
    total_predictions: Arc<AtomicU64>,
    total_latency_ms: Arc<AtomicU64>,
    total_errors: Arc<AtomicU64>,
    model_version: String,
    classifier: Arc<Mutex<FoodClassifier>>,
}

#[derive(Serialize)]
struct UploadResponse {
    filename: Option<String>,
    content_type: Option<String>,
    size_bytes: usize,
    width: u32,
    height: u32,
    predictions: Vec<Prediction>,
}

#[derive(Serialize)]
struct MetricsResponse {
    total_predictions: f64,
    avg_latency_ms: f64,
    total_errors: f64,
    model_version: String,
}

#[derive(Serialize)]
struct ModelInfoResponse {
    model: String,
    dataset: String,
    top1_accuracy: f64,
    top5_accuracy: f64,
    format: String,
    model_version: String,
}

async fn predict(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, StatusCode> {
    let start = Instant::now();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?
    {
        if field.name() == Some("image") {
            let filename = field.file_name().map(|name| name.to_string());
            let content_type = field.content_type().map(|ct| ct.to_string());
            let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;

            let image = load_image(&data).map_err(|_| StatusCode::BAD_REQUEST)?;
            let image = resize_and_crop(image);

            let width = image.width();
            let height = image.height();

            let tensor = image_to_tensor(&image);

            let predictions = state
                .classifier
                .lock()
                .await
                .predict_top5(tensor)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            let latency_ms = start.elapsed().as_millis() as u64;

            state.total_predictions.fetch_add(1, Ordering::Relaxed);
            state
                .total_latency_ms
                .fetch_add(latency_ms, Ordering::Relaxed);

            return Ok(Json(UploadResponse {
                filename,
                content_type,
                size_bytes: data.len(),
                width,
                height,
                predictions,
            }));
        }
    }

    Err(StatusCode::BAD_REQUEST)
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> Json<MetricsResponse> {
    let total_predictions = state.total_predictions.load(Ordering::Relaxed) as f64;
    let total_latency_ms = state.total_latency_ms.load(Ordering::Relaxed) as f64;
    let total_errors = state.total_errors.load(Ordering::Relaxed) as f64;

    let avg_latency_ms = if total_predictions > 0.0 {
        total_latency_ms / total_predictions
    } else {
        0.0
    };

    Json(MetricsResponse {
        total_predictions,
        avg_latency_ms,
        total_errors,
        model_version: state.model_version.clone(),
    })
}

async fn model_info_handler(State(state): State<Arc<AppState>>) -> Json<ModelInfoResponse> {
    Json(ModelInfoResponse {
        model: "MobileNetV3 Large".to_string(),
        dataset: "Food101".to_string(),
        top1_accuracy: 76.52,
        top5_accuracy: 93.76,
        format: "ONNX".to_string(),
        model_version: state.model_version.clone(),
    })
}

async fn health() -> &'static str {
    "Food101 Rust API is running"
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let model_path = "model/onnx/food101_mobilenetv3.onnx";
    let classes_path = "model/classes/food101_classes.txt";

    let classifier =
        FoodClassifier::new(model_path, classes_path).expect("Failed to load ONNX model/classes");

    println!("ONNX model loaded successfully");

    let state = Arc::new(AppState {
        classifier: Arc::new(Mutex::new(classifier)),
        total_predictions: Arc::new(AtomicU64::new(0)),
        total_latency_ms: Arc::new(AtomicU64::new(0)),
        total_errors: Arc::new(AtomicU64::new(0)),
        model_version: "v1.0.0".to_string(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/predict", post(predict))
        .route("/metrics", get(metrics_handler))
        .route("/model-info", get(model_info_handler))
        .with_state(state)
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(cors);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();

    println!("Backend running on http://localhost:8080");

    axum::serve(listener, app).await.unwrap();
}

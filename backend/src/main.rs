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
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

struct AppState {
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

async fn predict(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, StatusCode> {
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

async fn health() -> &'static str {
    "Food101 Rust API is running"
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let model_path = "../model/onnx/food101_mobilenetv3.onnx";
    let classes_path = "../model/classes/food101_classes.txt";

    let classifier =
        FoodClassifier::new(model_path, classes_path).expect("Failed to load ONNX model/classes");

    println!("ONNX model loaded successfully");

    let state = Arc::new(AppState {
        classifier: Arc::new(Mutex::new(classifier)),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/predict", post(predict))
        .with_state(state)
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();

    println!("Backend running on http://localhost:8080");

    axum::serve(listener, app).await.unwrap();
}

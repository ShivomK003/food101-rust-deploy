use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};

async fn health() -> &'static str {
    "Food101 Rust API is running"
}

async fn predict() -> &'static str {
    "Prediction endpoint coming soon"
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/predict", post(predict))
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .unwrap();

    println!("Backend running on http://localhost:8080");

    axum::serve(listener, app).await.unwrap();
}
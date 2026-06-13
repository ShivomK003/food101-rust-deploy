use anyhow::{Result, anyhow};
use ort::session::Session;
use ort::value::TensorRef;
use serde::Serialize;
use std::fs;

#[derive(Serialize)]
pub struct Prediction {
    pub class_name: String,
    pub class_index: usize,
    pub confidence: f32,
}

pub struct FoodClassifier {
    session: Session,
    classes: Vec<String>,
}

impl FoodClassifier {
    pub fn new(model_path: &str, classes_path: &str) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;

        let classes_text = fs::read_to_string(classes_path)?;
        let classes: Vec<String> = classes_text
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();

        if classes.len() != 101 {
            return Err(anyhow!(
                "Expected 101 Food101 classes, got {}",
                classes.len()
            ));
        }

        Ok(Self { session, classes })
    }

    pub fn predict(&mut self, tensor: Vec<f32>) -> Result<Vec<f32>> {
        let expected_len = 1 * 3 * 224 * 224;

        if tensor.len() != expected_len {
            return Err(anyhow!(
                "Invalid tensor length: expected {}, got {}",
                expected_len,
                tensor.len()
            ));
        }

        let input = TensorRef::from_array_view(([1_usize, 3, 224, 224], tensor.as_slice()))?;
        let outputs = self.session.run(ort::inputs![input])?;

        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(data.to_vec())
    }

    pub fn predict_top5(&mut self, tensor: Vec<f32>) -> Result<Vec<Prediction>> {
        let logits = self.predict(tensor)?;
        let probabilities = softmax(&logits);

        let mut indexed_probs: Vec<(usize, f32)> =
            probabilities.iter().copied().enumerate().collect();

        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let predictions = indexed_probs
            .into_iter()
            .take(5)
            .map(|(class_index, confidence)| Prediction {
                class_name: self.classes[class_index].clone(),
                class_index,
                confidence,
            })
            .collect();

        Ok(predictions)
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let exp_values: Vec<f32> = logits
        .iter()
        .map(|logit| (logit - max_logit).exp())
        .collect();

    let sum: f32 = exp_values.iter().sum();

    exp_values.iter().map(|value| value / sum).collect()
}

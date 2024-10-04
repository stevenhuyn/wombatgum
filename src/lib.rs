use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn run() -> PyResult<()> {
    let glove_vectors = load_glove_vectors("assets/glove.42B.300d.txt")?;

    // Example: Get vectors for two words and compute similarity
    let word1 = "king";
    let word2 = "queen";

    if let (Some(vec1), Some(vec2)) = (glove_vectors.get(word1), glove_vectors.get(word2)) {
        let similarity = cosine_similarity(vec1, vec2);
        println!("Cosine similarity between {} and {}: {}", word1, word2, similarity);
    } else {
        println!("Words not found in the GloVe dataset.");
    }

    Ok(())
}

#[pyfunction]
fn printer() -> PyResult<()> {
    println!("test");
    Python::with_gil(|py| {
        py.eval_bound("print('hello python')", None, None)?;
        Ok(())
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn glovers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(printer, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Result};
use ndarray::Array1;

/// A function to load GloVe vectors from a file into a HashMap.
fn load_glove_vectors(file_path: &str) -> Result<HashMap<String, Array1<f32>>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut word_vectors = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let mut tokens = line.split_whitespace();

        // First token is the word
        if let Some(word) = tokens.next() {
            // Collect the rest as the vector
            let vector: Array1<f32> = tokens
                .map(|token| token.parse::<f32>().expect("Could not parse float"))
                .collect::<Array1<f32>>();

            word_vectors.insert(word.to_string(), vector);
        }
    }

    Ok(word_vectors)
}

/// A function to compute cosine similarity between two vectors.
fn cosine_similarity(vec1: &Array1<f32>, vec2: &Array1<f32>) -> f32 {
    let dot_product = vec1.dot(vec2);
    let magnitude1 = vec1.dot(vec1).sqrt();
    let magnitude2 = vec2.dot(vec2).sqrt();
    dot_product / (magnitude1 * magnitude2)
}



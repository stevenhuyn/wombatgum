use ndarray::Array1;
use pyo3::prelude::*;
use rayon::iter::ParallelIterator;
use rayon::str::ParallelString;
use std::collections::HashMap;
use std::fs::read_to_string;
use std::io::Result;
use std::time::Instant;

/// A Python module implemented in Rust.
#[pymodule]
fn glovers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Glover>()?;
    m.add_class::<Foo>()?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(printer, m)?)?;
    Ok(())
}

#[pyclass]
struct Foo;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn printer() -> PyResult<()> {
    println!("printer final");
    Python::with_gil(|py| {
        py.eval_bound("print('hello python 2')", None, None)?;
        Ok(())
    })
}

// https://pyo3.rs/v0.22.3/class
#[pyclass]
struct Glover {
    vectors: HashMap<String, Array1<f32>>,
}

#[pymethods]
impl Glover {
    #[new]
    fn py_new() -> PyResult<Self> {
        let start = Instant::now();
        let vectors = Glover::load_glove_vectors("assets/glove.42B.300d.txt")?;
        let elapsed = start.elapsed();
        println!("Elapsed: {:.2}", elapsed.as_secs_f32());
        Ok(Self { vectors })
    }

    #[pyo3(signature = (word1, word2))]
    fn similar(&self, word1: &str, word2: &str) -> PyResult<PyObject> {
        if let (Some(vec1), Some(vec2)) = (self.vectors.get(word1), self.vectors.get(word2)) {
            let similarity = Glover::cosine_similarity(vec1, vec2);
            return Python::with_gil(|py| Ok(similarity.to_object(py)));
        }

        Python::with_gil(|py| Ok(py.None()))
    }
}

impl Glover {
    /// A function to load GloVe vectors from a file into a HashMap.
    fn load_glove_vectors(file_path: &str) -> Result<HashMap<String, Array1<f32>>> {
        let word_vectors: HashMap<String, Array1<f32>> = read_to_string(file_path)?
            .par_lines() // Read the lines from the file
            .filter_map(|line| {
                // Filter and map the lines in parallel
                let mut tokens = line.split_whitespace();
                if let Some(word) = tokens.next() {
                    let vector: Array1<f32> = tokens
                        .map(|token| token.parse::<f32>().expect("Could not parse float"))
                        .collect();
                    return Some((word.to_string(), vector));
                }
                None
            })
            .collect(); // Collect results into the HashMap

        Ok(word_vectors)
    }

    /// A function to compute cosine similarity between two vectors.
    fn cosine_similarity(vec1: &Array1<f32>, vec2: &Array1<f32>) -> f32 {
        let dot_product = vec1.dot(vec2);
        let magnitude1 = vec1.dot(vec1).sqrt();
        let magnitude2 = vec2.dot(vec2).sqrt();
        dot_product / (magnitude1 * magnitude2)
    }
}

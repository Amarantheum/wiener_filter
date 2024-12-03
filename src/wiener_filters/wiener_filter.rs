use nalgebra::{DVector, DMatrix};

/// A Weiner filter implementation.
/// See <a href="https://en.wikipedia.org/wiki/Wiener_filter" target="_blank">wikipedia</a> for more information.
pub struct WeinerFilter {
    coefficients: Vec<f64>,
}

impl WeinerFilter {
    /// Creates a new Weiner filter
    /// # Arguments
    /// * `observed_signals` - A vector of observed signals with the same length that are used to predict the desired signal
    /// * `desired_signal` - A desired signal with the same length as the observed signals
    /// ## **Note that the observed signals and the desired signal must be the same length or unexpected behavior will occur
    pub fn new(observed_signals: &Vec<&[f64]>, desired_signal: &[f64]) -> Self {
        let len = observed_signals[0].len();
        let mut correlation_matrix = DMatrix::<f64>::zeros(observed_signals.len(), observed_signals.len());
        for i in 0..observed_signals.len() {
            for j in 0..=i {
                let mut correlation = 0.0;
                for k in 0..len {
                    correlation += observed_signals[i][k] * observed_signals[j][k];
                }
                correlation /= len as f64;
                correlation_matrix[(i,j)] = correlation;
                correlation_matrix[(j,i)] = correlation;
            }
        }
        let mut correlation_vector = DVector::<f64>::zeros(observed_signals.len());
        for i in 0..observed_signals.len() {
            let mut correlation = 0.0;
            for k in 0..len {
                correlation += observed_signals[i][k] * desired_signal[k];
            }
            correlation /= len as f64;
            correlation_vector[i] = correlation;
        }

        let coefficients = correlation_matrix
            .lu()
            .solve(&correlation_vector)
            .expect("Singular matrix")
            .iter()
            .map(|x| *x)
            .collect();

        Self {
            coefficients,
        }
    }

    /// Filters the observed signals
    /// # Arguments
    /// * `observed_signals` - A vector of observed signals of the same length
    /// ## **Note that the observed signals must be the same length or unexpected behavior will occur
    /// # Returns
    /// A filtered signal with the same length as the observed signals
    pub fn filter(&self, observed_signals: &Vec<&[f64]>) -> Vec<f64> {
        let mut filtered = Vec::with_capacity(observed_signals.len());
        for i in 0..observed_signals[0].len() {
            let mut sum = 0.0;
            for j in 0..observed_signals.len() {
                sum += observed_signals[j][i] * self.coefficients[j];
            }
            filtered.push(sum);
        }
        filtered
    }

    /// Returns the coefficients of the Weiner filter
    pub fn coefficients(&self) -> &Vec<f64> {
        &self.coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use float_eq::assert_float_eq;

    #[test]
    fn test_wiener_filter() {
        const OBS_SIGNALS_SIZE: usize = 10;
        const OBS_LEN: usize = 1000;
        let mut rng = rand::thread_rng();
        let mut observed_signals = Vec::new();
        for _ in 0..OBS_SIGNALS_SIZE {
            let mut signal = Vec::new();
            for _ in 0..OBS_LEN {
                signal.push(rng.gen_range(-1.0..1.0));
            }
            observed_signals.push(signal);
        }
        let mut consts = Vec::new();
        for _ in 0..OBS_SIGNALS_SIZE {
            consts.push(rng.gen_range(-1.0..1.0));
        }
        let mut desired_signal = Vec::new();
        for i in 0..OBS_LEN {
            let mut sum = 0.0;
            for j in 0..OBS_SIGNALS_SIZE {
                sum += consts[j] * observed_signals[j][i];
            }
            desired_signal.push(sum);
        }
        let observed = observed_signals.iter().map(|x| x.as_slice()).collect();
        let wiener_filter = WeinerFilter::new(&observed, &desired_signal);
        for c in wiener_filter.coefficients.iter().zip(consts.iter()) {
            assert_float_eq!(c.0, c.1, abs <= 0.00000001);
        }

        let filtered = wiener_filter.filter(&observed);
        for i in 0..OBS_LEN {
            assert_float_eq!(filtered[i], desired_signal[i], abs <= 0.00000001);
        }
    }
}
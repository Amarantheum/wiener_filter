use nalgebra::{DVector, DMatrix};

/// An adaptive Weiner filter implementation that weights the observed signals with a geometric sequence.
pub struct GeometricWienerFilter {
    /// The geometric sequence weight
    alpha: f64,
    /// The coefficients of the Weiner filter
    coefficients: Vec<f64>,
    /// The covariance matrix of the observations
    observation_covariance: DMatrix<f64>,
    /// The covariance vector between the observations and the desired signal
    covariance_vector: DVector<f64>,
}

impl GeometricWienerFilter {
    /// Creates a new geometric Weiner filter
    /// # Arguments
    /// * `observed_signals` - A vector of observed signals of same length (>1) that are used to predict the desired signal
    /// * `desired_signal` - A desired signal with the same length as the observed signals
    /// * `alpha` - The geometric sequence weight (0 < alpha < 1 is recommended)
    /// ## **Note that the observed signals and the desired signal must be the same length or unexpected behavior will occur
    /// ## **Note that initial observation of length 1 results in a singular covariance matrix
    pub fn new(observed_signals: &Vec<&[f64]>, desired_signal: &[f64], alpha: f64) -> Self {
        let len = observed_signals[0].len();
        let mut correlation_matrix = DMatrix::<f64>::zeros(observed_signals.len(), observed_signals.len());
        for i in 0..observed_signals.len() {
            for j in 0..=i {
                let mut correlation = 0.0;
                for k in 0..len {
                    correlation += observed_signals[i][k] * observed_signals[j][k] * alpha.powi(len as i32 - k as i32 - 1);
                }
                correlation_matrix[(i,j)] = correlation;
                correlation_matrix[(j,i)] = correlation;
            }
        }
        let mut covariance_vector = DVector::<f64>::zeros(observed_signals.len());
        for i in 0..observed_signals.len() {
            let mut correlation = 0.0;
            for k in 0..len {
                correlation += observed_signals[i][k] * desired_signal[k] * alpha.powi(len as i32 - k as i32 - 1);
            }
            covariance_vector[i] = correlation;
        }

        let observation_covariance = correlation_matrix.clone();
        let coefficients = correlation_matrix
            .lu()
            .solve(&covariance_vector)
            .expect("Singular matrix")
            .iter()
            .map(|x| *x)
            .collect();

        
        GeometricWienerFilter {
            alpha,
            coefficients,
            observation_covariance,
            covariance_vector,
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

    /// Adds a single observation to the Weiner filter and updates the coefficients
    /// # Arguments
    /// * `observation` - A vector of observed values corresponding to the observed signals
    /// * `desired_signal` - The value of the desired signal
    pub fn add_observation(&mut self, observation: &Vec<f64>, desired_signal: f64) {
        debug_assert!(observation.len() == self.coefficients.len(), "Observation must have the same length as the coefficients");
        for i in 0..self.coefficients.len() {
            for j in 0..observation.len() {
                self.observation_covariance[(i,j)] = self.observation_covariance[(i,j)] * self.alpha + observation[i] * observation[j];
            }
            self.covariance_vector[i] = self.covariance_vector[i] * self.alpha + observation[i] * desired_signal;
        }
        self.coefficients = self.observation_covariance.clone()
            .lu()
            .solve(&self.covariance_vector)
            .expect("Singular matrix")
            .iter()
            .map(|x| *x)
            .collect();
    }

    /// Adds multiple observations to the Weiner filter and updates the coefficients
    /// # Arguments
    /// * `observations` - A vector of the observed signals
    /// * `desired_signal` - The desired signal with the same length as the observations
    /// ## **Note that the observations and the desired signal must be the same length or unexpected behavior will occur
    pub fn add_observations(&mut self, observations: &Vec<&[f64]>, desired_signal: &[f64]) {
        let len = observations[0].len();
        for i in 0..self.coefficients.len() {
            for j in 0..=i {
                let mut correlation = 0.0;
                for k in 0..len {
                    correlation += observations[i][k] * observations[j][k] * self.alpha.powi(len as i32 - k as i32 - 1);
                }
                let value = self.observation_covariance[(i,j)] * self.alpha.powi(len as i32) + correlation;
                self.observation_covariance[(i,j)] = value;
                self.observation_covariance[(j,i)] = value;
            }
        }
        for i in 0..self.coefficients.len() {
            let mut correlation = 0.0;
            for k in 0..len {
                correlation += observations[i][k] * desired_signal[k] * self.alpha.powi(len as i32 - k as i32 - 1);
            }
            let value = self.covariance_vector[i] * self.alpha.powi(len as i32) + correlation;
            self.covariance_vector[i] = value;
        }
        self.coefficients = self.observation_covariance.clone()
            .lu()
            .solve(&self.covariance_vector)
            .expect("Singular matrix")
            .iter()
            .map(|x| *x)
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use float_eq::assert_float_eq;

    #[test]
    fn test_basic_geometric_wiener_filter() {
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
        let alpha = 0.9;
        let geometric_wiener_filter = GeometricWienerFilter::new(&observed, &desired_signal, alpha);
        for c in geometric_wiener_filter.coefficients.iter().zip(consts.iter()) {
            assert_float_eq!(c.0, c.1, abs <= 0.00000001);
        }
    }
    

    #[test]
    fn test_add_observation_geometric_wiener_filter() {
        const OBS_SIGNALS_SIZE: usize = 10;
        const INIT_SIZE: usize = 10;
        const OBS_LEN: usize = 1000;
        let mut rng = rand::thread_rng();
        let mut observed_signals = Vec::new();
        for _ in 0..OBS_SIGNALS_SIZE {
            let mut signal = Vec::new();
            for _ in 0..OBS_LEN + INIT_SIZE {
                signal.push(rng.gen_range(-1.0..1.0));
            }
            observed_signals.push(signal);
        }
        let mut consts = Vec::new();
        for _ in 0..OBS_SIGNALS_SIZE {
            consts.push(rng.gen_range(-1.0..1.0));
        }
        let mut desired_signal = Vec::new();
        for i in 0..OBS_LEN + INIT_SIZE {
            let mut sum = 0.0;
            for j in 0..OBS_SIGNALS_SIZE {
                sum += consts[j] * observed_signals[j][i];
            }
            desired_signal.push(sum);
        }
        let alpha = 0.9;
        let init_observed = observed_signals.iter().map(|x| &x[0..INIT_SIZE]).collect();
        let mut geometric_wiener_filter = GeometricWienerFilter::new(&init_observed, &desired_signal[0..INIT_SIZE], alpha);

        for i in INIT_SIZE..OBS_LEN + INIT_SIZE {
            let observation = observed_signals.iter().map(|x| x[i]).collect();
            geometric_wiener_filter.add_observation(&observation, desired_signal[i]);
            for c in geometric_wiener_filter.coefficients.iter().zip(consts.iter()) {
                assert_float_eq!(c.0, c.1, abs <= 0.00000001);
            }
        }
    }

    #[test]
    fn test_add_basic_observation_geometric_wiener_filter() {
        let mut geometric_wiener_filter = GeometricWienerFilter::new(&vec![&[1.0]], &vec![2.0], 0.5);
        println!("{:#?}", geometric_wiener_filter.coefficients());
        geometric_wiener_filter.add_observation(&vec![1.0], 1.0);
        println!("{:#?}", geometric_wiener_filter.coefficients());
    }

    #[test]
    fn test_add_observations_geometric_wiener_filter() {
        const OBS_SIGNALS_SIZE: usize = 2;
        const INIT_SIZE: usize = 10;
        const OBS_LEN: usize = 1000;
        let mut rng = rand::thread_rng();
        let mut observations = Vec::new();
        let mut desired_signal = Vec::new();
        for _ in 0..OBS_SIGNALS_SIZE {
            let mut signal = Vec::new();
            for _ in 0..OBS_LEN + INIT_SIZE {
                signal.push(rng.gen_range(-1.0..1.0));
            }
            observations.push(signal);
        }
        for _ in 0..OBS_LEN + INIT_SIZE {
            desired_signal.push(rng.gen_range(-1.0..1.0));
        }
        let alpha = 0.9;
        let init_observed = observations.iter().map(|x| &x[0..INIT_SIZE]).collect();
        let init_desired = &desired_signal[0..INIT_SIZE];
        let mut filter1 = GeometricWienerFilter::new(&init_observed, init_desired, alpha);
        let mut filter2 = GeometricWienerFilter::new(&init_observed, init_desired, alpha);
        let mut filter3 = GeometricWienerFilter::new(&init_observed, init_desired, alpha);
        let mut filter4 = GeometricWienerFilter::new(&init_observed, init_desired, alpha);
        let mut iteration = 1;
        for i in INIT_SIZE..OBS_LEN + INIT_SIZE {
            let observation = observations.iter().map(|x| x[i]).collect();
            filter1.add_observation(&observation, desired_signal[i]);
            let observations_1 = observations.iter().map(|x| &x[i..i+1]).collect();
            filter2.add_observations(&observations_1, &desired_signal[i..i+1]);
            assert_float_eq!(filter1.coefficients()[0], filter2.coefficients()[0], abs <= 0.00000001);

            if iteration % 2 == 0 {
                let observations_2 = observations.iter().map(|x| &x[i-1..i+1]).collect();
                filter3.add_observations(&observations_2, &desired_signal[i-1..i+1]);
                assert_float_eq!(filter1.coefficients()[0], filter3.coefficients()[0], abs <= 0.00000001);
            }

            if iteration % 5 == 0 {
                let observations_3 = observations.iter().map(|x| &x[i-4..i+1]).collect();
                filter4.add_observations(&observations_3, &desired_signal[i-4..i+1]);
                assert_float_eq!(filter1.coefficients()[0], filter4.coefficients()[0], abs <= 0.00000001);
            }
            iteration += 1;
        }

    }
}
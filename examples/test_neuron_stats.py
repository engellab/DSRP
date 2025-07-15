import numpy as np
import logging
import matplotlib.pyplot as plt


from typing import Dict, Any

from dsr_project.neuron_stat import *
from dsr_project.DSR import *


import numpy as np
from typing import Dict, Any
import logging

# Set up logging for test output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_neuron_stats(DSR: Any, params_template: Dict[str, Dict], phi_values: np.ndarray, 
                     n_iterations: int = 10, time: int = 2000, num_trials: int = 100, 
                     waiting_time_cut: int = 2000, binsize: int = 200, shift: int = 50) -> np.ndarray:
    """Test neuron_stats function across a range of True_phi values.

    Args:
        DSR: Class for generating spike data (must have spike_generator method).
        params_template: Template dictionary for DSR parameters.
        phi_values: Array of True_phi values to test.
        n_iterations: Number of iterations per True_phi value (default: 10).
        time: Total time for spike generation (ms) (default: 2000).
        num_trials: Number of trials for spike generation (default: 100).
        waiting_time_cut: Maximum time window for analysis (ms) (default: 2000).
        binsize: Size of the small bin (ms) (default: 200).
        shift: Step size for sliding window (ms) (default: 50).

    Returns:
        np.ndarray: Array of shape (n_iterations, len(phi_values)) with computed phi values.

    Raises:
        ValueError: If inputs are invalid.
        NeuronStatsError: If neuron_stats or spike generation fails.
    """
    if not phi_values.size:
        raise ValueError("phi_values must be a non-empty array")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")

    # Initialize results array
    phis = np.zeros((n_iterations, len(phi_values)))

    for j, true_phi in enumerate(phi_values):
        # Update params with current True_phi
        params = params_template.copy()
        params['renewal_model']['params']['phi'] = true_phi

        logging.info(f"Testing True_phi = {true_phi:.1f}")
        for i in range(n_iterations):
            try:
                # Generate spike data
                spike_times, spike = DSR(**params).spike_generator(time=time, num_trials=num_trials)
                
                # Verify spike array
                if not isinstance(spike, np.ndarray) or spike.shape != (num_trials, time):
                    raise ValueError(f"Expected spike array of shape ({num_trials}, {time}), got {spike.shape}")

                # Compute phi
                phi_result = neuron_stats(
                    spike,
                    react_time=None,
                    waiting_time_cut=waiting_time_cut,
                    binsize=binsize,
                    shift=shift
                )
                
                # Store result (handle NeuronStatsResult if return_diagnostics=True)
                phis[i, j] = phi_result.phi if isinstance(phi_result, NeuronStatsResult) else phi_result
                
            except (NeuronStatsError, ValueError) as e:
                logging.error(f"Error in iteration {i+1} for True_phi = {true_phi:.1f}: {str(e)}")
                phis[i, j] = np.nan


    return phis


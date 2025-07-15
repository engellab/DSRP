
# 🧠 Doubly Stochastic Renewal Process 
A Set of Tools for Simulating and Analyzing Spiking Irregularity.

## 📘 References

If you use this package, please cite:

Aghamohammadi, C., Chandrasekaran, C., & Engel, T. A. (2024). A doubly stochastic renewal framework for partitioning
spiking variability. bioRxiv. [DOI: https://www.biorxiv.org/content/biorxiv/early/2024/02/23/2024.02.21.581457.full.pdf]


## Overview
The Doubly Stochastic Renewal Package is a Python toolkit designed for studying spiking irregularity in single-neuron spike trains. It offers two core components:
- **`DSR`**: A simulator for generating synthetic spike trains using a doubly stochastic renewal process.
- **`neuron_stats`**: An estimator for quantifying spiking irregularity (ϕ) from experimental or simulated spike data.

This package is ideal for modeling neural dynamics, validating computational models, and analyzing neuronal spiking patterns.

## 📦 Installation
The package requires **Python 3.8 or higher** and the following dependencies:
- `numpy>=1.21`
- `scipy>=1.7`
- `matplotlib>=3.4` (optional, for visualization with `show_spikes`)

We recommend using a virtual environment to avoid dependency conflicts:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install .
```

Or manually place the files in **src** folder into your Python path and import them.

---

## 📊 Component 1: DSR – **Spike Train Simulator**


**`DSR`  (Doubly Stochastic Renewal Process)**  allows the simulation of spike trains.  The  `DSR`  module contains a suite of methods for generating synthetic spike trains based on a doubly stochastic renewal process. The process is  defined  by a pair  $\{g(\cdot), \lambda(t)\}$  via a three-step algorithm for generating spike trains (see reference: Aghamohammadi, C., Chandrasekaran, C., & Engel, T. A.  _bioRxiv_  (2024)). Here,  $\lambda(t)$  is a stochastic process governing the instantaneous firing rate, and  $g(\cdot)$ is the renewal distribution.

The user can select from  among  many pre-defined firing rate models common in neuroscience  or provide their own user-defined firing rates. The user can also choose from several pre-defined renewal  distributions.

#### Firing Rate Model Functions

These callable functions return a time-varying firing rate array (in Hz) and can be passed to the DSR class for spike generation.

##### 1. `fix_constant_fr(parameters:{constant})`

Returns a fixed constant firing rate.

##### 2. `var_constant_fr(parameters:{min, max})`

Returns a constant firing rate chosen randomly between `min` and `max`.

##### 3. `fix_ramping(parameters:{start, end})`

Returns a linear ramping firing rate from `start` to `end`.

##### 4. `var_ramping(parameters:{start, end, width_var})`

Returns a ramping rate plus a single uniform noise term (`width_var * (0.5 - U[0,1])`).

##### 5. `DDM(parameters:{start, Boundry_low, Boundry_high, Drift, Diffusion})`

Generates a firing rate using the Drift-Diffusion Model. If the rate crosses the boundaries, it is truncated with `NaN`s.

##### 6. `DDAB(parameters:{start, Boundry_low, Boundry_high, Drift, Diffusion})`

Like DDM, but sticky boundaries: rate is clamped at the boundary instead of truncated.

##### 7. `OU_process(parameters:{start, sigma, mu, theta})`

Generates a firing rate via an Ornstein-Uhlenbeck process with quartic potential (i.e., restoring force scales cubically).

##### 8. `Feed(Fr_array)`

Pass-through function to use a user-specified rate array.


#### Pre-defined Renewal distribution

These functions generate inter-spike intervals (ISIs) in operational time from the pre-defined renewal distributions.

##### 1. `Clock_ISI_generator()`

Generates inter-spike intervals (ISIs) using a clock-like mechanism where ISIs are constant.

##### 2. `Poisson_ISI_generator()`

Generates inter-spike intervals (ISIs) using a Poisson distribution.

##### 3. `Gamma_generator(parameters:{phi})`

Generates inter-spike intervals (ISIs) using a Gamma distribution where phi (float) is the spiking irregularity for the Gamma distribution.


---

### Example 1 : Constant firing rate - Sub-Poisson

This example generates 20 trials of spike trains over 2000 ms with a constant firing rate of 10 Hz and a Gamma-distributed ISI (spiking irregularity phi=0.5). The show_spikes method displays a raster plot.

```python
from dsr_project import DSR, neuron_stat

params = {'firing_model':{"model": "fix_constant_fr", "params": {"constant":10}},
           'renewal_model':{"model": "Gamma_generator", "params": {"phi": 0.5}} } 

neuron = DSR(**params)
spike_times, spike = neuron.spike_generator(time=2000, num_trials=20)
neuron.show_spikes()
```

### Example 2 : Drift-Diffusion with Absorbing boundaries Model - Poisson

This example simulates spike trains using a Drift-Diffusion Model with absorbing boundaries and a Gamma renewal process (phi=0.3).

```python
from dsr_project import DSR, neuron_stat

params = {'firing_model':{"model": "DDAB", "params": {"start" : 10.0, "Boundry_low" : 1.0, "Boundry_high" : 20, "Drift" : 0, "Diffusion" : 20 }},
           'renewal_model':{"model": "Gamma_generator", "params": {"phi":0.3}} }  
neuron = DSR(**params)
spike_times, spike = neuron.spike_generator(time=2000, num_trials=20)

neuron.show_spikes()
```

### Example 3 : Using the firing rate of previous neuron as an input - Poisson

This example uses the firing rate from a previous DSR instance as input for a new simulation.

```python
from dsr_project import DSR, neuron_stat

params = {'firing_model':{"model": "Feed", "params": {"Fr_array": neuron.rates}},
           'renewal_model':{"model": "Gamma_generator", "params": {"phi":1}} }  
neuron = DSR(**params)
spike_times, spike = neuron.spike_generator(time=2000, num_trials=20)

neuron.show_spikes()
```

## 📊 Component 2: `neuron_stats` – **Spiking Irregularity Estimator (ϕ)**

The neuron_stats module estimates spiking irregularity (ϕ) from spike trains.

Parameters

-   spike: numpy.ndarray, binary spike array. Each element of the array represents 1 ms. A value of 1 indicates the presence of a spike, while 0 means no spike. 
    
-   react_time: float, optional reaction time (ms) to exclude from analysis (default: None).
    
-   waiting_time_cut: float, maximum time (ms) for ISI analysis (default: 2000).
    
-   binsize: float, bin size (ms) for irregularity estimation (default: 100).
    
-   shift: float, shift size (ms) for sliding window analysis (default: 50).


### Example

-  Synthetic Data Generation:
We generate synthetic spike data using:

1-The Drift-Diffusion Model as the firing rate model

2-The Gamma Distribution as the renewal distribution

-  Irregularity Estimation:

Using neuron_stat (a model-agnostic tool), we estimate the spiking irregularity of the neuron.

```python
from dsr_project import DSR, neuron_stat

params = {'firing_model':{"model": "DDAB", "params": {"start" : 30.0, "Boundry_low" : 20, "Boundry_high" : 30, "Drift" : 0, "Diffusion" : 20 }},
           'renewal_model':{"model": "Gamma_generator", "params": {"phi":0.5}} }  
neuron = DSR(**params)
spike_times, spike = neuron.spike_generator(time=2000, num_trials=20)


phi_result = neuron_stat.neuron_stats(
                    spike,
                    react_time=None,
                    waiting_time_cut=2000,
                    binsize=100,
                    shift=50
                )
```
### Examples

More examples can be find in examples folders:

-Example_spiking_DD_sticky_boundaries.ipynb

-Example_spiking_constant_FR.ipynb


##  License

This project is licensed under the MIT License. See the LICENSE file for details.


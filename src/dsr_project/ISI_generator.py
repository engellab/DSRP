import numpy as np

max_num_spikes_per_trial = 100

def Clock_ISI_generator():

    """
    Generates inter-spike intervals (ISIs) using a clock-like mechanism.

    Returns:
    - ISIs (numpy array): Array of ISIs with a constant value of 1.0
    """

    ISIs = np.ones(max_num_spikes_per_trial)

    first_ISI = np.array([np.random.rand()])

    ISIs = np.concatenate(( first_ISI, ISIs))

    return ISIs


################################################################################
################################################################################

def Poisson_ISI_generator():

    """
    Generates inter-spike intervals (ISIs) using a Poisson distribution.

    Returns:
    - ISIs (numpy array): Array of ISIs following a exponential distribution
    """

    ISIs = np.random.exponential(scale=1.0, size = max_num_spikes_per_trial)

    return ISIs

################################################################################
################################################################################

def Gamma_generator(phi = 0.5):

    """
    Generates inter-spike intervals (ISIs) using a Gamma distribution.

    Args:
    - phi (float): Spiking irregularity for the Gamma distribution

    Returns:
    - ISIs (numpy array): Array of ISIs following a Gamma distribution
    """

    ISIs = np.random.gamma(shape = 1.0/phi, scale = phi , size = max_num_spikes_per_trial)

    return ISIs

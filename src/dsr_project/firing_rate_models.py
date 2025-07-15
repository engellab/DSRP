
import numpy as np

max_time=5000

def fix_constant_fr(constant = 1.0):

    """
    Generates a fixed constant firing rate spike train.

    Args:
    - constant (float): Constant firing rate value

    Returns:
    - single_trial_fr (numpy array): Spike train with constant firing rate
    """

    single_trial_fr = constant * np.ones(max_time)

    return single_trial_fr


################################################################################
################################################################################

def var_constant_fr(min = 1.0, max=2.0):

    """
    Generates a variable constant firing rate spike train.

    Args:
    - min (float): Minimum firing rate value
    - max (float): Maximum firing rate value

    Returns:
    - single_trial_fr (numpy array): Spike train with variable constant firing rate
    """

    constant = min + (max- min) * np.random.rand()

    single_trial_fr = constant * np.ones(max_time)

    return single_trial_fr

################################################################################
################################################################################


def fix_ramping(start = 1.0, end = 2.0):

    """
    Generates a fixed ramping firing rate spike train.

    Args:
    - start (float): Starting firing rate value
    - end (float): Ending firing rate value

    Returns:
    - single_trial_fr (numpy array): Spike train with fixed ramping firing rate
    """

    single_trial_fr = start + ((end-start)/max_time) * np.arange(max_time)

    return single_trial_fr


################################################################################
################################################################################


def var_ramping(start = 1.0, end = 2.0, width_var = 0.5):

    """
    Generates a variable ramping firing rate spike train.

    Args:
    - start (float): Starting firing rate value
    - end (float): Ending firing rate value
    - width_var (float): Width variation for randomness

    Returns:
    - single_trial_fr (numpy array): Spike train with variable ramping firing rate
    """

    single_trial_fr = start + ((end-start)/max_time) * np.arange(max_time) + width_var * ( 0.5 - np.random.rand() )

    return single_trial_fr


################################################################################
################################################################################


def DDM(start = 10.0, Boundry_low = 2.0, Boundry_high = 2.0, Drift = 0.5, Diffusion = 0.2 ):

    """
    Generates a spike train using the Drift-Diffusion Model.

    Args:
    - start (float): Starting firing rate value
    - Boundry_low (float): Lower bound for spiking threshold
    - Boundry_high (float): Upper bound for spiking threshold
    - Drift (float): Drift coefficient
    - Diffusion (float): Diffusion coefficient

    Returns:
    - single_trial_fr (numpy array): Spike train generated using DDM
    """


    steps = np.random.rand(max_time)-0.5
    noise = Diffusion * np.sqrt(0.001) * np.cumsum(steps)
    single_trial_fr = start + 0.001 * Drift * np.arange(max_time) + noise

    if len(np.where(single_trial_fr > Boundry_high)[0]) > 0:
        single_trial_fr[ np.where(single_trial_fr > Boundry_high)[0][0]: ] = np.nan
    if len(np.where(single_trial_fr < Boundry_low)[0]) > 0:
        single_trial_fr[ np.where(single_trial_fr < Boundry_low)[0][0]: ] = np.nan

    return single_trial_fr

################################################################################
################################################################################

def Feed( Fr_array = np.ones((10,10)) ):

    """
    Dummy function returning provided firing rate array.

    Args:
    - Fr_array (numpy array): Firing rate array

    Returns:
    - Fr_array (numpy array): Provided firing rate array
    """

    return Fr_array

################################################################################
################################################################################

def OU_process(start = 10.0, sigma = 2.0, mu = 10.0, theta = 1.0):

    """
    Generates a spike train using the Ornstein-Uhlenbeck process with quartic potential.

    Args:
    - start (float): Starting firing rate value
    - sigma (float): Sigma coefficient
    - mu (float): Mu coefficient
    - theta (float): Theta coefficient

    Returns:
    - single_trial_fr (numpy array): Spike train generated using OU process with quartic potential.
    """

    dt = 0.001
    sqrtdt = np.sqrt(dt)
    sigma_bis = sigma * np.sqrt(2.0*theta)

    single_trial_fr = np.zeros(max_time)
    single_trial_fr[0] = start

    for i in range(max_time-1):

        # if single_trial_fr[i]  > BC_high:
        #     single_trial_fr[i + 1] = BC_high - 0.1
        # elif single_trial_fr[i]  < BC_low:
        #     single_trial_fr[i + 1] = BC_low + 0.1
        # else:
        #     single_trial_fr[i + 1] = single_trial_fr[i] - dt * theta*(single_trial_fr[i] - mu) + sigma_bis * sqrtdt * np.random.randn()


        single_trial_fr[i + 1] = single_trial_fr[i] - dt * theta*(single_trial_fr[i] - mu)**3 + sigma_bis * sqrtdt * np.random.randn()


    return single_trial_fr

################################################################################
################################################################################


def DDAB(start = 10.0, Boundry_low = 2.0, Boundry_high = 20, Drift = 0.5, Diffusion = 0.2 ):

    """
    Generates a spike train using the Drift-Diffusion with sticky Boundaries Model.

    Args:
    - start (float): Starting firing rate value
    - Boundry_low (float): Lower bound for spiking threshold
    - Boundry_high (float): Upper bound for spiking threshold
    - Drift (float): Drift coefficient
    - Diffusion (float): Diffusion coefficient

    Returns:
    - single_trial_fr (numpy array): Spike train generated using DDAB model
    """

    steps = np.random.rand(max_time)-0.5
    noise = Diffusion * np.sqrt(0.001) * np.cumsum(steps)
    single_trial_fr = start + 0.001 * Drift * np.arange(max_time) + noise

    if len(np.where(single_trial_fr > Boundry_high)[0]) > 0:
        single_trial_fr[ np.where(single_trial_fr > Boundry_high)[0][0]: ] = Boundry_high
    if len(np.where(single_trial_fr < Boundry_low)[0]) > 0:
        single_trial_fr[ np.where(single_trial_fr < Boundry_low)[0][0]: ] = Boundry_low

    return single_trial_fr

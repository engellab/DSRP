
import numpy as np

def time_rescaling(rate, raw_ISIs, time):

    """
    Performs time rescaling to map spikes in operational time to spike in real times based on the given firing rate.

    Args:
    - rate (numpy array): Firing rate profile over time
    - raw_ISIs (numpy array): Raw inter-spike intervals
    - time (int): Length of spike train

    Returns:
    - spike (numpy array): Spike times after time rescaling
    """

    raw_spike = ( 1000*np.cumsum(raw_ISIs) ).astype(int)

    rat_cum = ( np.cumsum(rate) ).astype(int)

    raw_spike = raw_spike[ raw_spike < np.max(rat_cum)]

    spike = []

    for i in np.arange(len(raw_spike)):
        spike.append(np.where(raw_spike[i] < rat_cum )[0][0])

    spike = np.array(spike)
    spike = spike[ spike < time]

    return spike

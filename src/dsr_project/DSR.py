from . import firing_rate_models
from . import ISI_generator
from . import time_rescaling
import numpy as np
import matplotlib.pyplot as plt

class DSR:

    """
    Doubly Stochastic Renewal Process Object
    Attributes:
    - firing_model: Stochastic process for firing rate
    - renewal_model: Function to generate inter-spike intervals (ISIs)
    """

    def __init__(self, firing_model, renewal_model):

        # Initialize with firing and renewal models

        self.firing_model = firing_model
        self.renewal_model = renewal_model

    def spike_generator(self, time, num_trials):

        """
        Generates spike trains using specified firing and renewal models.
        Args:
        - time: Length of spike trains
        - num_trials: Number of spike train repetitions
        Returns:
        - spike_times: List of spike times for each trial
        - spikes: Binary spike trains matrix
        """

        # Options for firing rate models

        firing_model_map = {
                'fix_constant_fr': firing_rate_models.fix_constant_fr,    # The firing rate is constant in time and does not change from trial to trial
                'var_constant_fr': firing_rate_models.var_constant_fr,    # The firing rate is constant in time but it changes from trial to trial
                'fix_ramping': firing_rate_models.fix_ramping,            # The firing rate linearly ramps in time and it does not change from trial to trial
                'var_ramping': firing_rate_models.var_ramping,            # The firing rate linearly ramps in time and the ofset changes stochasticly from trial to trial
                'DDM':firing_rate_models.DDM,                             # The firing rate comes from the Drift-Diffusion process. When it reaches the boundaries the trial will end.
                'Feed':firing_rate_models.Feed,                           # The feed your firing rate rates of interests as a matrix (trial*time)
                'OU_process':firing_rate_models.OU_process,               # The firing rate comes from the OU process with quartic potential
                'DDAB':firing_rate_models.DDAB                            # The firing rate comes from the Drift-Diffusion process with sticky boundaries.
        }

        # Options for renwal function

        renewal_model_map = {
            'Clock_ISI_generator': ISI_generator.Clock_ISI_generator,      # clock like spiking. The renewal distribution is delta functions at 1 seconds.
            'Poisson_ISI_generator': ISI_generator.Poisson_ISI_generator,  # Poisson spiking. Matching this with any firing rate model generates an inhomogenous Poisson process.
            'Gamma_generator': ISI_generator.Gamma_generator               # Gamma distribution with one parametrer $\phi$ controling spiking irregularity.
        }


        fr_model = firing_model_map[self.firing_model['model']]
        fr_model_params = self.firing_model['params']

        renewal_model = renewal_model_map[self.renewal_model['model']]
        renewal_model_params = self.renewal_model['params']

        spike_times = []
        spikes = np.zeros(shape = (num_trials,time) )
        rates = np.zeros(shape = (num_trials,time) )

        for trial_num in range(num_trials):

            rate = fr_model(**fr_model_params)
            if self.firing_model['model'] == 'Feed':
                rate = rate[trial_num,:]

            raw_ISIs = renewal_model(**renewal_model_params)

            spike_time = time_rescaling.time_rescaling(rate, raw_ISIs, time)

            spike_times.append(spike_time)

            spikes[trial_num,spike_time] = 1
            rates[trial_num,:] = rate[0:time]


        self.spike_times = spike_times
        self.spikes = spikes
        self.rates = rates

        #self.spike_times = spike_time

        return spike_times,spikes


    def show_spikes(self):

        """Visualizes spike trains and firing rates."""


        width_cm = 7*3
        height_cm = 3*3

        # Convert centimeters to inches
        width_inch = width_cm / 2.54
        height_inch = height_cm / 2.54

        #plt.rcParams["figure.figsize"] = (14,7)
        fig, ax = plt.subplots(1,2, figsize=(width_inch, height_inch), constrained_layout=True)

        color_arr = ['#D3A1D3' , '#BF73B9' , '#995691', '#724470', '#583658', '#8E77FF', '#7C67E5', '#675EC2', '#52559E', '#393C75']


        plt.subplot(1,2,1)
        x=[]
        y=[]

        spike = self.spikes
        rates = self.rates


        for trial_num in range(spike.shape[0]):

            x = np.where(spike[trial_num])[0]
            y = trial_num * np.ones( len(np.where(spike[trial_num])[0]))

            for sp_num in range(x.shape[0]):
                plt.plot([x[sp_num],x[sp_num]], [y[sp_num],y[sp_num]+0.7],color=color_arr[np.mod(trial_num,10)], linewidth = 0.5 )


        plt.xlabel("Time (ms)" )
        plt.ylabel("Trial" )

        plt.xticks([0, int(rates.shape[1]/2), rates.shape[1]], [0, int(rates.shape[1]/2), rates.shape[1]])
        plt.xticks(fontname = 'Helvetica' ,fontsize = 12)

        # for spine in plt.gca().spines.values():
        #     spine.set_visible(False)

        plt.tick_params(top=False, left= False, right=False, labelleft = False, labelbottom='on')
        plt.gca().xaxis.set_tick_params(width=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        asp_ratio = 0.6 * rates.shape[1]/rates.shape[0]


        plt.gca().set_aspect(asp_ratio)

        #######################################################################

        plt.subplot(1,2,2)


        for trial_num in range(rates.shape[0]):
            plt.plot(rates[trial_num], color=color_arr[np.mod(trial_num,10)], linewidth = 0.5)

        plt.xlabel("Time (ms)" )
        plt.ylabel("Firing rate (Hz)" )

        rate_wNan = rates[~np.isnan(rates)]
        plt.xticks([0, int(rates.shape[1]/2), rates.shape[1]], [0, int(rates.shape[1]/2), rates.shape[1]])
        plt.yticks([int( np.floor(np.min(rate_wNan)) )-1, int( np.floor(np.mean(rate_wNan)) ) , int( np.floor(np.max(rate_wNan)+1) )], [int( np.floor(np.min(rate_wNan))-1 ), int( np.floor(np.mean(rate_wNan)) ) , int( np.floor(np.max(rate_wNan)+1) )])

        plt.xticks(fontname = 'Helvetica' ,fontsize = 12)
        plt.yticks(fontname = 'Helvetica' ,fontsize = 12)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        asp_ratio = 0.6 * rates.shape[1]/( np.floor(np.max(rate_wNan)) - np.floor(np.min(rate_wNan)) + 2 )

        plt.gca().set_aspect(asp_ratio)

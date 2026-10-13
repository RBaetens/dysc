import numpy as np
import matplotlib.pyplot as plt

import spectrogram_tools as st
import simulation_tools as simt


#########################################
# Damped Duffing oscillator spectrogram #
#########################################

# # Damped Duffing equations
# def damped_duffing(x):
#     return np.array([x[1], x[0]-x[0]**3-0.025*x[1]])

# DuffingSystem = simt.ContinuousNonlinearTimeInvariant(f=damped_duffing, n=2, x0=np.ones(2))

# # Generate data
# fs = 100
# dt = 1/fs
# npts = 10000
# T = (npts-1)*dt
# t, x = DuffingSystem.run(T, num=npts)

# # Plot
# start_plot = 0
# len_plot = 10000
# plot_idxs = np.arange(start_plot, start_plot+len_plot, dtype=int)

# fig, ax = plt.subplots()
# ax.scatter(x[0, plot_idxs], x[1, plot_idxs], c=t[plot_idxs])
# fig.tight_layout()
# plt.show()

# # Spectrogram
# _, _ = st.wavelet_spectrogram(x[0], fs, "cmor8-1", freq_bounds=[0.05, 0.3], freq_scale="log10", plot_scale="linear", plot_hw_ratio=0.4)
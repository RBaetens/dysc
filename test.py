import numpy as np
import matplotlib.pyplot as plt

import spectrogram_tools as st
import simulation_tools as simt
import pysindy as ps


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


############################################
# Damped Duffing oscillator identification #
############################################

# Damped Duffing equations
def damped_duffing(x):
    return np.array([x[1], x[0]-x[0]**3-0.025*x[1]])

DuffingSystem = simt.ContinuousNonlinearTimeInvariant(f=damped_duffing, n=2, v=None, x0=np.ones(2))

# Generate data
fs = 100
dt = 1/fs
npts = 10000
T = (npts-1)*dt
t, x = DuffingSystem.run(T, num=npts)

# Add noise to data: more like measurement noise, less like disturbance
v = simt.StationaryGaussianWhiteNoise(n=2, mu=np.zeros(2), cov=np.diag(np.ones(2)*0.0001), seed=83)
noise = v.sample(k=npts)
x = x+noise

# Plot
start_plot = 0
len_plot = 10000
plot_idxs = np.arange(start_plot, start_plot+len_plot, dtype=int)

fig, ax = plt.subplots()
ax.scatter(x[0, plot_idxs], x[1, plot_idxs], c=t[plot_idxs])
fig.tight_layout()
plt.show()

# Initiate model
differentiation_method = ps.FiniteDifference(order=4)
feature_library = ps.PolynomialLibrary(degree=3)
optimizer = ps.STLSQ(threshold=0.02)

nonlinear_model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
)

# Fit model
nonlinear_model.fit(np.transpose(x), dt, feature_names=["x", "y"])

# Show the identified model
nonlinear_model.print()
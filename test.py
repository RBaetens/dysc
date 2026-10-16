import numpy as np
import matplotlib.pyplot as plt

import spectrogram_tools as st
import simulation_tools as simt
import pysindy as ps
from scipy.signal import butter, sosfiltfilt
from scipy.integrate import cumulative_trapezoid


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

# # Damped Duffing equations
# def damped_duffing(x):
#     return np.array([x[1], x[0]-x[0]**3-0.025*x[1]])

# DuffingSystem = simt.ContinuousNonlinearTimeInvariant(f=damped_duffing, n=2, v=None, x0=np.ones(2))

# # Generate data
# fs = 100
# dt = 1/fs
# npts = 10000
# T = (npts-1)*dt
# t, x = DuffingSystem.run(T, num=npts)

# # Add noise to data: more like measurement noise, less like disturbance
# v = simt.StationaryGaussianWhiteNoise(n=2, mu=np.zeros(2), cov=np.diag(np.ones(2)*0.0001), seed=83)
# noise = v.sample(k=npts)
# x = x+noise

# # Plot
# start_plot = 0
# len_plot = 10000
# plot_idxs = np.arange(start_plot, start_plot+len_plot, dtype=int)

# fig, ax = plt.subplots()
# ax.scatter(x[0, plot_idxs], x[1, plot_idxs], c=t[plot_idxs])
# fig.tight_layout()
# plt.show()

# # Initiate model
# differentiation_method = ps.FiniteDifference(order=4)
# feature_library = ps.PolynomialLibrary(degree=3)
# optimizer = ps.STLSQ(threshold=0.02)

# nonlinear_model = ps.SINDy(
#     differentiation_method=differentiation_method,
#     feature_library=feature_library,
#     optimizer=optimizer,
# )

# # Fit model
# nonlinear_model.fit(np.transpose(x), dt, feature_names=["x", "y"])

# # Show the identified model
# nonlinear_model.print()


###########################
# NES and TMD spectrogram #
###########################

# nes_arr = np.loadtxt("C:/Users/robaeten/Downloads/NileNES.csv", delimiter=",")
# tmd_arr = np.loadtxt("C:/Users/robaeten/Downloads/NileTMD.csv", delimiter=",")

# print(nes_arr.shape)
# print(tmd_arr.shape)

# # NES
# t_axis_nes = nes_arr[:, 0]
# T_nes = t_axis_nes[-1]-t_axis_nes[0]
# npts_nes = t_axis_nes.size
# dt_nes = T_nes/(npts_nes-1)
# fs_nes = 1/dt_nes

# x_nes = nes_arr[:, 1] - np.linspace(nes_arr[0, 1], nes_arr[-1, 1], num=npts_nes)

# fig, ax = plt.subplots()
# ax.plot(t_axis_nes, x_nes)
# plt.show()

# _, _, _ = st.gabor_spectrogram(x_nes, fs_nes, 10, 1, hop=100, over_sampling=4, freq_bounds=[0.5, 4.5], plot_scale="linear", colorbar=True)
# _, _, _ = st.gabor_spectrogram(x_nes, fs_nes, 10, 1, hop=100, over_sampling=4, freq_bounds=[0.5, 4.5], plot_scale="log", colorbar=True)
# # _, _ = st.wavelet_spectrogram(x_nes, fs_nes, "cmor10-5", freq_bounds=[0.5, 4.5], freq_scale="linear", plot_scale="linear", plot_hw_ratio=0.4)
# # _, _ = st.wavelet_spectrogram(x_nes, fs_nes, "cmor10-5", freq_bounds=[0.5, 4.5], freq_scale="linear", plot_scale="log", plot_hw_ratio=0.4)

# # TMD
# t_axis_tmd = tmd_arr[:, 0]
# T_tmd = t_axis_tmd[-1]-t_axis_tmd[0]
# npts_tmd = t_axis_tmd.size
# dt_tmd = T_tmd/(npts_tmd-1)
# fs_tmd = 1/dt_tmd

# col_tmd = 3
# x_tmd = tmd_arr[:, col_tmd]

# fig, ax = plt.subplots()
# ax.plot(t_axis_tmd, x_tmd)
# plt.show()

# _, _, _ = st.gabor_spectrogram(x_tmd, fs_tmd, 10, 1, hop=100, over_sampling=4, freq_bounds=[0.5, 4.5], plot_scale="linear", colorbar=True)
# _, _, _ = st.gabor_spectrogram(x_tmd, fs_tmd, 10, 1, hop=100, over_sampling=4, freq_bounds=[0.5, 4.5], plot_scale="log", colorbar=True)
# # _, _ = st.wavelet_spectrogram(x_tmd, fs_tmd, "cmor10-5", freq_bounds=[0.5, 4.5], freq_scale="linear", plot_scale="linear", plot_hw_ratio=0.4)
# # _, _ = st.wavelet_spectrogram(x_tmd, fs_tmd, "cmor10-5", freq_bounds=[0.5, 4.5], freq_scale="linear", plot_scale="log", plot_hw_ratio=0.4)


######################
# NES identification #
######################

# Load data
data_arr = np.loadtxt("data/vibrations/Sweep_HardeningNES.csv", delimiter=",", skiprows=1)
t_axis, ddu, ddx = data_arr[:, 0], data_arr[:, 1], data_arr[:, 2] # Time and absolute coordinates
ddz = ddx-ddu # Relative coordinates

# Calculate state variables from measurements
npts = t_axis.size
dt = (t_axis[-1]-t_axis[0])/(npts-1)
fs = 1/dt
print(npts)
sos_integration = butter(2, [1, 50], btype="bandpass", analog=False, output="sos", fs=fs)

# ddx_filt = sosfiltfilt(sos_integration, ddx)
# ddz_filt = sosfiltfilt(sos_integration, ddz)
# ddu_filt = sosfiltfilt(sos_integration, ddu)
ddx_filt = ddx
ddz_filt = ddz
ddu_filt = ddu
ddx_filt_nb = ddx_filt - np.mean(ddx_filt)
ddz_filt_nb = ddz_filt - np.mean(ddz_filt)
ddu_filt_nb = ddu_filt - np.mean(ddu_filt)
dx = np.cumsum(ddx_filt_nb)*dt
dz = np.cumsum(ddz_filt_nb)*dt
du = np.cumsum(ddu_filt_nb)*dt

dx_filt = sosfiltfilt(sos_integration, dx)
dz_filt = sosfiltfilt(sos_integration, dz)
du_filt = sosfiltfilt(sos_integration, du)
dx_filt_nb = dx_filt - np.mean(dx_filt)
dz_filt_nb = dz_filt - np.mean(dz_filt)
du_filt_nb = du_filt - np.mean(du_filt)
x = np.cumsum(dx_filt_nb)*dt
z = np.cumsum(dz_filt_nb)*dt
u = np.cumsum(du_filt_nb)*dt

x_filt = sosfiltfilt(sos_integration, x)
z_filt = sosfiltfilt(sos_integration, z)
u_filt = sosfiltfilt(sos_integration, u)
x_filt_nb = x_filt - np.mean(x_filt)
z_filt_nb = z_filt - np.mean(z_filt)
u_filt_nb = u_filt - np.mean(u_filt)

# # Plot data
# fig, axs = plt.subplots(nrows=2, figsize=(10, 13))

# axs[0].plot(t_axis, ddu, label="raw")
# axs[0].plot(t_axis, ddu_filt_nb, label="filt")
# axs[0].set_title("Base acceleration")
# axs[0].set_xlabel("Time [s]")
# axs[0].set_ylabel("Acceleration [m/s$^2$]")
# axs[0].legend()
# axs[0].grid()

# axs[1].plot(t_axis, ddz, label="raw")
# axs[1].plot(t_axis, ddz_filt_nb, label="filt")
# axs[1].set_title("Absorber mass acceleration")
# axs[1].set_xlabel("Time [s]")
# axs[1].set_ylabel("Acceleration [m/s$^2$]")
# axs[1].legend()
# axs[1].grid()

# fig.tight_layout()
# plt.show()

# # Plot
# fig, axs = plt.subplots(nrows=4, figsize=(10, 13))

# # plot_start = 25*fs
# # plot_time = 5*fs
# # plot_idxs = np.arange(plot_start, plot_start+plot_time, dtype=int)
# plot_start = 0
# plot_time = npts
# plot_idxs = np.arange(plot_start, plot_start+plot_time, dtype=int)

# axs[0].plot(t_axis[plot_idxs], z_filt_nb[plot_idxs])
# axs[0].plot(t_axis[plot_idxs], np.zeros(npts)[plot_idxs])
# axs[0].set_title("Absorber position")
# axs[0].set_xlabel("Time [s]")
# axs[0].set_ylabel("Position [m]")

# axs[1].plot(t_axis[plot_idxs], u_filt_nb[plot_idxs])
# axs[1].plot(t_axis[plot_idxs], np.zeros(npts)[plot_idxs])
# axs[1].set_title("Base position")
# axs[1].set_xlabel("Time [s]")
# axs[1].set_ylabel("Position [m]")

# axs[2].plot(t_axis[plot_idxs], dz_filt_nb[plot_idxs])
# axs[2].plot(t_axis[plot_idxs], np.zeros(npts)[plot_idxs])
# axs[2].set_title("Absorber velocity")
# axs[2].set_xlabel("Time [s]")
# axs[2].set_ylabel("Velocity [m/s]")

# axs[3].plot(t_axis[plot_idxs], du_filt_nb[plot_idxs])
# axs[3].plot(t_axis[plot_idxs], np.zeros(npts)[plot_idxs])
# axs[3].set_title("Base velocity")
# axs[3].set_xlabel("Time [s]")
# axs[3].set_ylabel("Velocity [m/s]")

# fig.tight_layout()
# plt.show()

# Compose data matrices
clip = 10000
train_idxs = np.arange(clip, npts-clip)
X = np.concat((z_filt_nb.reshape((-1, 1)), dz_filt_nb.reshape((-1, 1))), axis=1)[train_idxs]
DX = np.concat((dz_filt_nb.reshape((-1, 1)), ddz_filt_nb.reshape((-1, 1))), axis=1)[train_idxs]
U = (ddu_filt_nb).reshape((-1, 1))[train_idxs]

# Initiate model
differentiation_method = ps.FiniteDifference(order=4)
feature_library = ps.PolynomialLibrary(degree=4)
optimizer = ps.SR3(reg_weight_lam=10000, normalize_columns=True, max_iter=1000, regularizer="L0", relax_coeff_nu=1)

nonlinear_model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
)

nonlinear_model.fit(X, t_axis[train_idxs], x_dot=DX, u=U)

nonlinear_model.print()
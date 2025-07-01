import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Union, Optional
import nbformat
from scipy.signal import ShortTimeFFT, stft
from scipy.signal.windows import gaussian
import pywt  # PyWavelets


# Load data and inspect dataframe
full_path = "path to parquet file"

df = pd.read_parquet(full_path)

file_name = full_path.split("/")[-1]

df.info()
df.describe()
df.columns

# Select a single column for analysis and drop NaN
subset_col = "ay_uc_device"
df_subset = df[[subset_col]].dropna()

df_subset.info()


x = df_subset.values
fs = 20
dt = 1 / fs
t = np.arange(len(x)) / fs


### Continuous Wavelet Transform (CWT)

wavelet_name = "cmor1.5-1.0"
# 'cmorB-C' pattern, where B = bandwidth, C = center frequency
# cmor = complex Morlet wavelet

# Check fc of wavelet, wavelet’s center (dimensionless) frequency:

f_c = pywt.central_frequency("cmor1.5-1.0")

# Determine scale range by setting min and max freqs:

f_min = 0.5  # lower freq of interest (Hz)
f_max = 5  # upper freq of interest (Hz) set to Nyqist limit, fs / 2,  or lower

scale_max = f_c / (f_min * dt)  # corresponds to f_min
scale_min = f_c / (f_max * dt)  # corresponds to f_max

scales = np.arange(scale_min, scale_max)

coef, freqs = pywt.cwt(x, scales, wavelet_name, sampling_period=1 / fs)

# coef shape: (36, 194252, 1)
# freqs shape: (36, )

coef_magnitude = np.abs(coef)
extent = [t[0], t[-1], freqs[-1], freqs[0]]

start_s = 30 * 60
int_s = 5
plt.figure(figsize=(10, 6))
plt.imshow(
    coef_magnitude,
    extent=extent,
    aspect="auto",
    cmap="jet",
    norm=colors.LogNorm(vmin=np.abs(coef).min() + 1e-12, vmax=np.abs(coef).max()),
)
plt.title(f"Continuous Wavelet Transform - {subset_col}")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.xlim(start_s, start_s + int_s)
plt.colorbar(label="Magnitude")
plt.show()

start_s = 3000
int_s = 10
vmax = np.percentile(coef_magnitude, 99)  # ignore the top 1%
plt.figure(figsize=(10, 6))
plt.imshow(coef_magnitude, extent=extent, aspect="auto", cmap="jet", vmin=0, vmax=vmax)
plt.title(f"Continuous Wavelet Transform - {subset_col}")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.xlim(start_s, start_s + int_s)
plt.colorbar(label="Magnitude")
plt.show()

### Discrete Wavelet Transform (DWT) ###

# -------------------------------------------------------
# Step 1: Prepare Data
# -------------------------------------------------------

x = df_subset.values
fs = 20.0  # sampling frequency (Hz)

# -------------------------------------------------------
# Step 2: Choose Wavelet and Decomposition Level
# -------------------------------------------------------
wavelet = "db4"  # Daubechies 4 is a common choice
max_level = 8  # A reasonable starting point

# Check max safe level:

max_level_safe = pywt.dwt_max_level(
    data_len=len(x), filter_len=pywt.Wavelet(wavelet).dec_len
)
print("Recommended maximum level:", max_level_safe)

# -------------------------------------------------------
# Step 3: Perform the Discrete Wavelet Transform
# -------------------------------------------------------
# wavedec returns a list of coefficients: [cA_L, cD_L, cD_(L-1), ..., cD_1]
coeffs = pywt.wavedec(x, wavelet=wavelet, level=max_level)

# coeffs[0] = approximation at level max_level
# coeffs[1:] = detail coefficients from level max_level down to level 1

# -------------------------------------------------------
# Step 4: Visualize Approximation & Details
# -------------------------------------------------------
# For a quick overview, let's plot each set of coefficients.
plt.figure(figsize=(12, 8))
num_plots = len(coeffs)  # 1 approximation + max_level details

# Plot the approximation (lowest-frequency component)
ax = plt.subplot(num_plots, 1, 1)
ax.plot(coeffs[0], color="blue")
ax.set_title(f"Approximation (Level {max_level})")

# Plot detail coefficients
for i, cD in enumerate(coeffs[1:], start=1):
    ax = plt.subplot(num_plots, 1, i + 1)
    ax.plot(cD, color="red")
    ax.set_title(f"Detail (Level {max_level - i + 1})")

plt.tight_layout()
plt.show()

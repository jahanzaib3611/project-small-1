# Student B (ID: 23036048)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Loading dataset
dataset_path = "airline4.csv"
data = pd.read_csv(dataset_path)
data['Date'] = pd.to_datetime(data['Date'])

# Extracting year, month, and day
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Fourier Transform
time_series = data['Number'].values
n = len(time_series)
frequencies = np.fft.fftfreq(n)
fft_values = fft(time_series)

# Calculate total passengers in 2022 : X
data_2022 = data[data['Year'] == 2022]
total_passengers_2022 = data_2022['Number'].sum()

# Power Spectrum & value of Y 
power_spectrum = np.abs(fft_values[:n // 2])**2
non_zero_frequencies = frequencies[:n // 2][frequencies[:n // 2] != 0]
power_spectrum_non_zero = power_spectrum[frequencies[:n // 2] != 0]
main_period_length = 1 / non_zero_frequencies[np.argmax(power_spectrum_non_zero)]

# Plottings
plt.figure(figsize=(10, 6))
plt.bar(range(1, 13), data.groupby('Month')['Number'].mean(), color='skyblue', label='Monthly Average Passengers')
plt.xlabel('Month')
plt.ylabel('Passengers (in thousands)')
plt.title('Monthly Passenger Analysis\nStudent ID: 23036048')
plt.legend()
plt.savefig('plot1.png')

plt.figure(figsize=(10, 6))
plt.plot(non_zero_frequencies, power_spectrum_non_zero, label='Power Spectrum')
plt.xlabel('Frequency (1/day)')
plt.ylabel('Power')
plt.title('Power Spectrum\nStudent ID: 23036048')
plt.legend()
plt.savefig('plot2.png')


print(f"Value of X (Total Passengers in 2022): {total_passengers_2022} (in thousands)")
print(f"Value of Y (Main Period Length): {main_period_length:.2f} days")


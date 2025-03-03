import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt


# Define parameters for a sine wave
frequency = 1  
sampling_rate = 100  
duration = 1 
time = np.arange(0, duration, 1/sampling_rate)

# Generate the sine wave
sine_wave = np.sin(2 * np.pi * frequency * time)

# Define noise
noise_mean = 0
noise_std = 0.2  # Adjust for desired noise level

# use np.random to generate random offsets
noise = np.random.normal(noise_mean, noise_std, size=len(time))

# Add the noise to sine wave
noisy_sine_wave = sine_wave + noise

#Define random seed
np.random.seed(42)

#reshape from 1d -> 2d arrays for fit and predict
X = sine_wave.reshape(-1,1)
y = noisy_sine_wave.reshape(-1,1)

#create a pair of Decision Tree Regressor:
reg_tree_3 = DecisionTreeRegressor(max_depth=3,random_state=42)
reg_tree_3.fit(X,y)
reg_tree_5 = DecisionTreeRegressor(max_depth=5,random_state=42)
reg_tree_5.fit(X,y)

#use the tree to predict values:
y_pred_3 = reg_tree_3.predict(X)
y_pred_5 = reg_tree_5.predict(X)


plt.figure(figsize=(10, 6))
plt.scatter(time, noisy_sine_wave, label='Sine Wave with Noise')
plt.plot(time, y_pred_3,label='Predicted values, max_depth = 3', color='red')
plt.plot(time,y_pred_5, label ='Predicted values, max_depth=5',color='green' )
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Decision Tree Regressor with Different Max Depths')
plt.legend()
plt.grid(True)
plt.show()
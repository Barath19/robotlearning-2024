import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Function to extract scalar summary data from TensorBoard logs
def extract_scalar_summary(logdir):
    data = {'steps': [], 'rewards': []}
    for event in tf.compat.v1.train.summary_iterator(logdir):
        for value in event.summary.value:
            if value.HasField('simple_value') and value.tag == "Reward / Instantaneous reward (mean)":
                data['steps'].append(event.step)
                data['rewards'].append(value.simple_value)
    return data

# Function to calculate mean and variance using a sliding window
def calculate_mean_and_variance(data, window_size):
    mean_values = []
    variance_values = []
    for i in range(len(data['rewards']) - window_size + 1):
        window_data = data['rewards'][i:i+window_size]
        mean_values.append(sum(window_data) / window_size)
        variance_values.append(sum((x - mean_values[-1]) ** 2 for x in window_data) / window_size)
    return mean_values, variance_values

# Function to apply moving average to data
def moving_average(data, window_size):
    smoothed_data = []
    for i in range(len(data) - window_size + 1):
        smoothed_data.append(sum(data[i:i+window_size]) / window_size)
    return smoothed_data

# Function to plot scalar data using Matplotlib
def plot_scalar_data_with_variance(data, mean_values, variance_values, xlabel='', ylabel=''):
    trimmed_steps = data['steps'][:len(mean_values)]  # Trim the steps to match the length of the smoothed data
    plt.plot(trimmed_steps, mean_values, label='Mean')
    plt.fill_between(trimmed_steps, [mean - var for mean, var in zip(mean_values, variance_values)], [mean + var for mean, var in zip(mean_values, variance_values)], alpha=0.2)

# Extract scalar summary data
# Path to the directory containing TensorBoard logs
log_dir_ppo = 'logs/ppo_kinova'
scalar_data_ppo = extract_scalar_summary(log_dir_ppo)

log_dir_sac = 'logs/sac_kinova'
scalar_data_sac = extract_scalar_summary(log_dir_sac)

log_dir_sac = 'logs/td3_kinova'
scalar_data_td3 = extract_scalar_summary(log_dir_sac)

# Calculate mean and variance
window_size = 10
mean_values_ppo, variance_values_ppo = calculate_mean_and_variance(scalar_data_ppo, window_size)
mean_values_sac, variance_values_sac = calculate_mean_and_variance(scalar_data_sac, window_size)
mean_values_td3, variance_values_td3 = calculate_mean_and_variance(scalar_data_td3, window_size)

# Plot scalar data with mean and variance
plt.figure(figsize=(8, 8))  # Adjust figure size for square shape
plot_scalar_data_with_variance(scalar_data_ppo, mean_values_ppo, variance_values_ppo, xlabel='Steps', ylabel='Reward')
plot_scalar_data_with_variance(scalar_data_sac, mean_values_sac, variance_values_sac, xlabel='Steps', ylabel='Reward')
#plot_scalar_data_with_variance(scalar_data_td3, mean_values_td3, variance_values_td3, xlabel='Steps', ylabel='Reward')

title = 'Kinova Cabinet Opening Task'
plt.title(title)
plt.legend(['PPO', 'SAC'], loc='lower right')
plt.grid(True)
plt.xlabel('Steps')
plt.ylabel('Reward')

# Save the plot as a square SVG file
plt.savefig('kinova_reward_plot.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig('kinova_reward_plot.png', format='png', dpi=300, bbox_inches='tight')

plt.show()

#!/usr/bin/env python3
"""
Script to plot gas usage vs evaluations required for training operations.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Data from the user
evaluations_required = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
avg_gas_per_trainer = [357299, 325199, 316175, 443761, 573672, 1182237, 1770336, 4683455, 7329986, 26754517]



# Convert to numpy arrays for easier manipulation
x_data = np.array(evaluations_required)
y_data = np.array(avg_gas_per_trainer)

# Define a polynomial function for curve fitting (quadratic)
def polynomial_func(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the curve
popt, pcov = curve_fit(polynomial_func, x_data, y_data)

# Generate smooth curve for plotting
x_smooth = np.linspace(min(x_data), max(x_data), 100)
y_smooth = polynomial_func(x_smooth, *popt)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot the data points
plt.scatter(evaluations_required, avg_gas_per_trainer, color='blue', s=100, alpha=0.7, label='Data Points', zorder=5)

# Add a line connecting the points
plt.plot(evaluations_required, avg_gas_per_trainer, color='red', alpha=0.5, linewidth=2, label='Linear Interpolation')

# Plot the fitted curve
plt.plot(x_smooth, y_smooth, color='green', linewidth=3, alpha=0.8, label=f'Fitted Curve (Quadratic)')

# Customize the plot
plt.xlabel('Evaluations Required', fontsize=12, fontweight='bold')
plt.ylabel('Average Gas Per Trainer', fontsize=12, fontweight='bold')
plt.title('Gas Usage vs Evaluations Required', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Format y-axis to show gas in thousands/millions
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x < 1000000 else f'{x/1000000:.1f}M'))

# Add annotations for each data point (removed)

# Add legend
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('/Users/haseebsaeed/codes/rizemind/gas_usage_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("Plot saved as 'gas_usage_plot.png'")
print("\nData Summary:")
print("Evaluations Required | Avg Gas Per Trainer")
print("-" * 40)
for eval_req, gas in zip(evaluations_required, avg_gas_per_trainer):
    print(f"{eval_req:>18} | {gas:>18,}")

print(f"\nFitted Curve Parameters (y = ax² + bx + c):")
print(f"a = {popt[0]:.2f}")
print(f"b = {popt[1]:.2f}")
print(f"c = {popt[2]:.2f}")

# Calculate R-squared
y_pred = polynomial_func(x_data, *popt)
ss_res = np.sum((y_data - y_pred) ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared = {r_squared:.4f}")

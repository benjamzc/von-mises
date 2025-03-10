#!/usr/bin/env python
"""
Script to generate a visualization of the von Mises distribution
with different concentration parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import i0  # Modified Bessel function of first kind, order 0

# Set plotting style
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5)

def von_mises_pdf(x, mu, kappa):
    """von Mises probability density function."""
    return np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))

# Create figure
plt.figure(figsize=(12, 8))

# Define values for x-axis (angles from -pi to pi)
x = np.linspace(-np.pi, np.pi, 1000)

# Plot von Mises distributions with different concentration parameters
concentrations = [0.5, 2.0, 5.0, 10.0, 20.0]
colors = ['#FF9999', '#66B2FF', '#99CC99', '#FFCC99', '#CC99CC']

for i, kappa in enumerate(concentrations):
    # Calculate PDF values
    pdf = von_mises_pdf(x, 0, kappa)
    
    # Plot
    plt.plot(x, pdf, color=colors[i], linewidth=3, label=f'κ = {kappa}')

# Add vertical line at mean direction (μ = 0)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Mean direction (μ = 0)')

# Customize plot
plt.title('Von Mises Distribution with Different Concentration Parameters', fontsize=16)
plt.xlabel('Angle (radians)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.xlim(-np.pi, np.pi)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')

# Save figure
plt.tight_layout()
plt.savefig('figures/von_mises_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved von_mises_distribution.png")

plt.close() 
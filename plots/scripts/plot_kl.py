import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_smoothed_timeseries_wandb_style(
    x_values,
    y_values,
    smoothing_window_size=50,
    outlier_detection_window_size=20,  # Window for calculating rolling stats for outliers
    outlier_std_factor=3,  # Number of std devs to define an outlier
    original_color_hex="#a9d6e5",  # Light blue
    smoothed_color_hex="#014f86",  # Darker blue
    figsize=(12, 6),
    title="Metric Over Training Steps",
    xlabel="train/global_step",
    ylabel="Value",
):
    """
    Plots original time series data along with its moving average,
    applying outlier clipping and using a WandB-inspired aesthetic.

    Args:
        x_values (np.ndarray): X-coordinates of the data points.
        y_values (np.ndarray): Y-coordinates of the data points.
        smoothing_window_size (int): Window size for the moving average.
        outlier_detection_window_size (int): Window size for rolling mean/std
                                             used in outlier detection.
        outlier_std_factor (float): Data points further than this many rolling
                                    standard deviations from the rolling mean
                                    are considered outliers and clipped.
        original_color_hex (str): Hex color for the original data line.
        smoothed_color_hex (str): Hex color for the smoothed data line.
        figsize (tuple): Figure size for the plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if len(y_values) < smoothing_window_size:
        print(
            f"Warning: Data length ({len(y_values)}) is less than smoothing_window_size ({smoothing_window_size}). "
            "Smoothing might not be effective or possible."
        )
        # Plot raw data if smoothing is not feasible
        if len(y_values) == 0:
            print("No data to plot.")
            return

        plt.figure(figsize=figsize)
        sns.set_style("whitegrid")
        plt.plot(
            x_values,
            y_values,
            color=original_color_hex,
            alpha=0.7,
            linewidth=1.5,
            label="Original Data (Insufficient for smoothing)",
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        sns.despine()  # Removes top and right spines
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # --- 1. Outlier Handling (Clipping) ---
    # Use pandas for convenient rolling calculations
    y_series = pd.Series(y_values)

    # Calculate rolling mean and std for outlier detection
    # min_periods=1 ensures we get values even at the start of the series
    rolling_mean = y_series.rolling(
        window=outlier_detection_window_size, center=True, min_periods=1
    ).mean()
    rolling_std = y_series.rolling(
        window=outlier_detection_window_size, center=True, min_periods=1
    ).std()

    # Define outlier bounds
    # Fill NaN in rolling_std with a large value or 0 to avoid issues if std is 0 for flat regions
    # If std is 0, any deviation is an outlier, which might be too aggressive.
    # A small epsilon can be added to rolling_std to prevent division by zero or overly sensitive clipping.
    # For simplicity, we'll fill NaN std with a high value (less clipping) or 0 (if mean is stable).
    # Let's fill NaN std with the global std if available, or a small positive number.
    global_std_fallback = y_series.std() if not y_series.std() == 0 else 1.0
    rolling_std_filled = rolling_std.fillna(global_std_fallback)

    lower_bound = rolling_mean - outlier_std_factor * rolling_std_filled
    upper_bound = rolling_mean + outlier_std_factor * rolling_std_filled

    # Clip the original y_values
    y_clipped = np.clip(y_values, lower_bound, upper_bound)

    # --- 2. Calculate Moving Average ---
    # The 'valid' mode means the output is shorter, where the window fully overlaps.
    # The result has length max(M,N) - min(M,N) + 1.
    # If N is len(y_clipped) and M is smoothing_window_size, then len = N - M + 1.
    if len(y_clipped) >= smoothing_window_size:
        # Create a weights array for the moving average
        weights = np.ones(smoothing_window_size) / smoothing_window_size
        y_smoothed = np.convolve(y_clipped, weights, mode="valid")

        # Adjust x_values for the smoothed line. The first smoothed point corresponds
        # to the data window ending at index (smoothing_window_size - 1).
        # So, the x-coordinate is typically taken as the x-value at the end or center of that window.
        # For mode='valid', the x-values start from x_values[smoothing_window_size - 1]
        x_smoothed = x_values[smoothing_window_size - 1 :]

        # Alternative: center the smoothed value in the window
        # half_window = (smoothing_window_size - 1) // 2
        # x_smoothed = x_values[half_window : -half_window]
        # if smoothing_window_size % 2 == 0: # Adjust if even window size for perfect centering
        #    x_smoothed = x_values[half_window : -(half_window+1)]
        # Ensure y_smoothed and x_smoothed have compatible lengths if using alternative centering.
        # The np.convolve 'valid' mode aligns well with x_values[smoothing_window_size-1:].

    else:
        # Not enough data points to compute a valid moving average with the given window
        y_smoothed = np.array([])
        x_smoothed = np.array([])
        print(
            f"Warning: Not enough data points ({len(y_clipped)}) for smoothing window ({smoothing_window_size}). Smoothed line will not be plotted."
        )

    # --- 3. Plotting ---
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")  # Seaborn style for aesthetics

    # Plot original (clipped) data
    plt.plot(
        x_values,
        y_clipped,
        color=original_color_hex,
        alpha=0.4,
        linewidth=1.5,
        label="Original Data (Outliers Clipped)",
    )

    # Plot smoothed data
    if len(y_smoothed) > 0:
        plt.plot(
            x_smoothed,
            y_smoothed,
            color=smoothed_color_hex,
            linewidth=2.5,
            label=f"Moving Average (Window: {smoothing_window_size})",
        )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    sns.despine()  # Removes top and right spines for a cleaner look like WandB
    plt.legend(fontsize=10)
    plt.tight_layout()  # Adjusts plot to prevent labels from overlapping
    plt.show()


# --- Generate Sample Data (mimicking the provided image's characteristics) ---
np.random.seed(42)  # for reproducibility
num_points = 280
x = np.arange(num_points)

# Base trend (quadratic + sine wave for some variation)
y_trend = 0.0001 * (x - 50) ** 2 - 0.5 + 0.3 * np.sin(x / 30)
# Noise
y_noise = np.random.normal(0, 0.4, num_points)  # General noise
# Add some sparse, larger "outliers"
num_outliers = 15
outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
outlier_magnitudes = np.random.uniform(-2, 2, num_outliers)  # Make them significant
y_signal = y_trend + y_noise
y_signal[outlier_indices] += outlier_magnitudes

# Make the trend stronger towards the end
growth_factor = np.linspace(
    1, 2.5, num_points
)  # Create a stronger upward push at the end
y_signal = (y_trend + y_noise) * growth_factor
y_signal[outlier_indices] += (
    outlier_magnitudes * 1.5
)  # Make outliers more pronounced with growth


# --- Call the plotting function ---
plot_smoothed_timeseries_wandb_style(
    x_values=x,
    y_values=y_signal,
    smoothing_window_size=50,
    outlier_detection_window_size=30,  # Window for calculating rolling stats for outliers
    outlier_std_factor=2.5,  # Clip points beyond 2.5 std devs from rolling mean
    title="Training Rewards Over Steps",
    xlabel="Global Step",
    ylabel="Cumulative Reward",
)

# Example with fewer points (to test edge cases)
# plot_smoothed_timeseries_wandb_style(
#     x_values=x[:40],
#     y_values=y_signal[:40],
#     smoothing_window_size=50,
#     title="Short Training Rewards",
#     xlabel="Global Step",
#     ylabel="Cumulative Reward"
# )

# Example with very few points
# plot_smoothed_timeseries_wandb_style(
#     x_values=x[:5],
#     y_values=y_signal[:5],
#     smoothing_window_size=10,
#     title="Very Short Training Rewards",
#     xlabel="Global Step",
#     ylabel="Cumulative Reward"
# )

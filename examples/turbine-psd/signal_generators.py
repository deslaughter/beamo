# Signal generation utilities

import numpy as np
import scipy
import plotly.graph_objects as go
import plotly.subplots as sp
import os


def genWhiteNoise(dt, duration, seed=None):
    """
    Generates white zero-mean Gaussian noise signal with a specified sampling time and duration.
    Variance is scaled by the sampling time to ensure correct power spectral density.
        Parameters
        ----------
        dt : float
            Sampling time in seconds.
        duration : float
            Duration of the signal in seconds.
        seed : int, optional
            Random seed for reproducibility. If None, a random seed is used.
        Returns
        -------
        np.ndarray
            Array of white noise samples.
    """
    if seed is not None:
        np.random.seed(seed)
    num_samples = int(duration / dt)
    return np.random.normal(0, 1, num_samples)  # * np.sqrt(dt)


def generate_white_noise_frequency_domain(dt, duration, seed=None):
    """
    Generates a white noise signal by creating a flat spectrum in the frequency domain
    and performing an inverse FFT.

    Args:
        num_samples (int): The number of samples in the generated signal.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        numpy.ndarray: The white noise signal as a NumPy array (time domain).
    """

    if seed is not None:
        np.random.seed(seed)

    num_samples = int(duration / dt)
    sampling_rate = 1 / dt

    # Create an array of frequencies
    freqs = np.fft.rfftfreq(num_samples, 1 / sampling_rate)

    # Create a flat magnitude spectrum (uniform in frequency)
    # We use rfft, so we only need positive frequencies and the DC component.
    magnitude_spectrum = np.ones_like(freqs)

    # Generate random phases for each frequency component (between -pi and pi)
    random_phases = np.random.uniform(-np.pi, np.pi, len(freqs))

    # Combine magnitude and phase to create the complex frequency spectrum
    complex_spectrum = magnitude_spectrum * np.exp(1j * random_phases)

    # Perform inverse real FFT to get the time-domain signal
    white_noise_signal = np.fft.irfft(complex_spectrum, n=num_samples)

    # scaling to be between -1 and 1
    # white_noise_signal /= np.max(np.abs(white_noise_signal))

    return white_noise_signal


def lowPassFilter(signal, cutoff_freq, dt):
    """
    Applies a low-pass filter to the input signal using scipy.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to be filtered.
    cutoff_freq : float
        Cutoff frequency for the low-pass filter in Hz.
    dt : float
        Time step of the input signal in seconds.

    Returns
    -------
    np.ndarray
        Low-pass filtered signal.
    """
    nyquist_freq = 0.5 / dt
    if cutoff_freq >= nyquist_freq:
        raise ValueError("Cutoff frequency must be less than Nyquist frequency.")

    # Design Butterworth low-pass filter
    b, a = scipy.signal.butter(4, cutoff_freq / nyquist_freq, btype="low")

    signal = scipy.signal.filtfilt(b, a, signal)

    # scale the signal to be between -1 and 1
    signal /= np.max(np.abs(signal))

    return signal


def convert2binary(signal, threshold=0.0):
    """
    Converts a continuous signal to a binary signal based on a threshold.
    If the signal is greater than the threshold, set to 1, else set to -1.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to be converted.
    threshold : float, optional
        Threshold value for binary conversion. Default is 0.0.

    Returns
    -------
    np.ndarray
        Binary signal where values above the threshold are set to 1, and others to -1.
    """
    return np.where(signal > threshold, 1, -1)


def genRBSignal(dt, duration, cutoff_freq=0.1, seed=None, plot=False):
    """
    Generates a random binary signal with specified sampling time and duration using
    white noise generation, low-pass filtering, and binary conversion.

    Plotting uses Plotly.

    Parameters
    ----------
    dt : float
        Sampling time in seconds.
    duration : float
        Duration of the signal in seconds.
    cutoff_freq : float, optional
        Cutoff frequency for the low-pass filter in Hz. Default is 0.1 Hz.
    seed : int, optional
        Random seed for reproducibility. If None, a random seed is used.

    Returns
    -------
    np.ndarray
        Array of random binary signal samples.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate white noise
    # noise = genWhiteNoise(dt, duration, seed=seed)
    noise = generate_white_noise_frequency_domain(dt, duration, seed=seed)

    # Apply low-pass filter
    filtered = lowPassFilter(noise, cutoff_freq, dt)

    # Convert to binary
    binary_signal = convert2binary(filtered)

    # if plot:
    # Create time vector
    time = np.arange(0, duration, dt)

    # Create a Plotly figure
    fig = sp.make_subplots(rows=2, cols=1, subplot_titles=("Generated Signals", "FFT"))
    fig.add_trace(
        go.Scatter(x=time, y=noise, mode="lines", name="White Noise"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=filtered, mode="lines", name="Filtered Signal"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time, y=binary_signal, mode="lines", name="Binary Signal"),
        row=1,
        col=1,
    )

    # Compute FFT of all signals
    fft_noise = np.fft.fft(noise)
    fft_filtered = np.fft.fft(filtered)
    fft_binary = np.fft.fft(binary_signal)
    freq = np.fft.fftfreq(len(fft_binary), dt)

    # Get only the positive frequencies for plotting
    positive_freq_idxs = np.where(freq > 0)
    positive_freq = freq[positive_freq_idxs]
    positive_fft_noise = np.abs(fft_noise)[positive_freq_idxs]
    positive_fft_filtered = np.abs(fft_filtered)[positive_freq_idxs]
    positive_fft_binary = np.abs(fft_binary)[positive_freq_idxs]

    # Plot FFTs (positive frequencies only)
    fig.add_trace(
        go.Scatter(
            x=positive_freq,
            y=positive_fft_noise,
            mode="lines",
            name="FFT of White Noise",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=positive_freq,
            y=positive_fft_filtered,
            mode="lines",
            name="FFT of Filtered Signal",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=positive_freq,
            y=positive_fft_binary,
            mode="lines",
            name="FFT of Binary Signal",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(type="log", row=2, col=1, title="Frequency (Hz)")
    fig.update_yaxes(type="log", row=2, col=1, title="Magnitude")

    # adding marker for the cutoff frequency
    fig.add_vline(
        x=cutoff_freq, line_width=2, line_dash="dash", line_color="red", row=2, col=1
    )

    if plot:
        fig.show()
    else:
        os.makedirs("RBSplots", exist_ok=True)
        fig.write_html(f"RBSplots/generated_signal_{seed}.html")

    return binary_signal


if __name__ == "__main__":
    # Example usage
    dt = 0.01  # Sampling time in seconds
    duration = 1000  # Duration in seconds
    cutoff_freq = 10  # Cutoff frequency in Hz
    seeds = np.random.randint(0, 45487823, 12)  # Random seed for reproducibility
    signals = np.array(
        [
            lowPassFilter(
                generate_white_noise_frequency_domain(dt, duration, seed=seed),
                cutoff_freq,
                dt,
            )
            for seed in seeds
        ]
    ).T
    np.savetxt("generated_signal.txt", signals)

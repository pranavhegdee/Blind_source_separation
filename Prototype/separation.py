import streamlit as st
import torch
import torchaudio
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import sounddevice as sd
import wavio
from asteroid.models import ConvTasNet
from asteroid.utils import tensors_to_device
import tempfile
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import librosa
import seaborn as sns

@st.cache_resource
def load_model():
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    model.eval()
    return model

def compute_dft_manual(x, N=None):
    """Manual DFT computation for educational purposes"""
    if N is None:
        N = len(x)
    
    X = np.zeros(N, dtype=complex)
    n = np.arange(N)
    
    for k in range(N):
        X[k] = np.sum(x * np.exp(-1j * 2 * np.pi * k * n / N))
    
    return X

def linear_convolution(x, h):
    """Manual linear convolution implementation"""
    M = len(x)
    N = len(h)
    y = np.zeros(M + N - 1)
    
    for n in range(M + N - 1):
        for k in range(M):
            if 0 <= n - k < N:
                y[n] += x[k] * h[n - k]
    
    return y

def circular_convolution(x, h, N):
    """Manual circular convolution implementation"""
    y = np.zeros(N)
    
    for n in range(N):
        for k in range(N):
            y[n] += x[k] * h[(n - k) % N]
    
    return y

def compute_signal_features(audio, sr, ref_signal=None):
    """Compute comprehensive signal processing features"""
    features = {}
    features['Mean'] = np.mean(audio)
    features['Variance'] = np.var(audio)
    features['RMS'] = np.sqrt(np.mean(audio**2))
    features['Peak'] = np.max(np.abs(audio))
    features['Crest Factor'] = features['Peak'] / (features['RMS'] + 1e-10)
    features['Zero Crossing Rate'] = np.sum(np.diff(np.sign(audio)) != 0) / (2 * len(audio))
    
    N = len(audio)
    freqs = fftfreq(N, 1/sr)
    X = fft(audio)
    magnitude = np.abs(X)
    
    features['Spectral Centroid'] = np.sum(freqs[:N//2] * magnitude[:N//2]) / (np.sum(magnitude[:N//2]) + 1e-10)
    features['Spectral Bandwidth'] = np.sqrt(np.sum(((freqs[:N//2] - features['Spectral Centroid'])**2) * magnitude[:N//2]) / (np.sum(magnitude[:N//2]) + 1e-10))
    features['Spectral Rolloff'] = freqs[np.where(np.cumsum(magnitude[:N//2]) >= 0.85 * np.sum(magnitude[:N//2]))[0][0]] if np.sum(magnitude[:N//2]) > 0 else 0
    features['Total Energy'] = np.sum(audio**2)
    features['Average Power'] = features['Total Energy'] / len(audio)
    features['Skewness'] = np.mean(((audio - features['Mean']) / np.sqrt(features['Variance'] + 1e-10))**3)
    features['Kurtosis'] = np.mean(((audio - features['Mean']) / np.sqrt(features['Variance'] + 1e-10))**4)
    
    # SNR: Estimate noise as low-amplitude components
    noise_floor = np.percentile(np.abs(audio), 10)
    signal_power = np.mean(audio**2)
    noise_power = np.mean((audio[np.abs(audio) < noise_floor])**2) if np.any(np.abs(audio) < noise_floor) else 1e-10
    features['SNR_dB'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # SDR: For separated sources, use ref_signal if provided
    if ref_signal is not None:
        distortion = audio - ref_signal
        signal_power = np.mean(audio**2)
        distortion_power = np.mean(distortion**2) + 1e-10
        features['SDR_dB'] = 10 * np.log10(signal_power / distortion_power)
    
    # Cross-Correlation Peak
    if ref_signal is not None:
        norm_factor = np.sqrt(np.sum(audio*2) * np.sum(ref_signal*2)) + 1e-10
        cross_corr = signal.correlate(audio, ref_signal, mode='full') / norm_factor
        features['Cross_Correlation_Peak'] = np.max(np.abs(cross_corr))
    
    return features

def compute_chroma_features(audio, sr):
    """Compute chroma features from audio signal"""
    try:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_var = np.var(chroma, axis=1)
        
        chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        chroma_features = {}
        for i, note in enumerate(chroma_names):
            chroma_features[f'Chroma_{note}_mean'] = chroma_mean[i]
            chroma_features[f'Chroma_{note}_var'] = chroma_var[i]
        
        return chroma_features, chroma
    except Exception as e:
        st.warning(f"Chroma analysis failed: {str(e)}")
        return {}, np.array([])

def compute_energy_ratio(signal1, signal2):
    """Compute energy ratio between two signals"""
    energy1 = np.sum(signal1**2)
    energy2 = np.sum(signal2**2)
    return energy1 / (energy2 + 1e-10)

def plot_spectrogram_and_chroma(audio, sr, title="Spectrogram and Chroma Analysis"):
    """Plot spectrogram and chroma features"""
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    
    time = np.linspace(0, len(audio)/sr, len(audio))
    axs[0].plot(time, audio)
    axs[0].set_title(f"{title} - Waveform", fontweight='bold')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True, alpha=0.3)
    
    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024, noverlap=512)
    im1 = axs[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    axs[1].set_title(f"{title} - Spectrogram", fontweight='bold')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axs[1], label='Power (dB)')
    
    try:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)
        chroma_time = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)
        chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        im2 = axs[2].pcolormesh(chroma_time, np.arange(12), chroma, shading='gouraud', cmap='plasma')
        axs[2].set_title(f"{title} - Chroma Features", fontweight='bold')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Chroma')
        axs[2].set_yticks(np.arange(12))
        axs[2].set_yticklabels(chroma_names)
        plt.colorbar(im2, ax=axs[2], label='Chroma Intensity')
    except Exception as e:
        axs[2].text(0.5, 0.5, f'Chroma analysis not available: {str(e)}', 
                   transform=axs[2].transAxes, ha='center', va='center')
        axs[2].set_title(f"{title} - Chroma Features (Not Available)", fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_dft_analysis(audio, sr, title="DFT Analysis"):
    """Comprehensive DFT/FFT analysis plots"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    N = len(audio)
    freqs = fftfreq(N, 1/sr)
    X_fft = fft(audio)
    
    subset_size = min(512, N)
    audio_subset = audio[:subset_size]
    X_dft_manual = compute_dft_manual(audio_subset)
    freqs_subset = fftfreq(subset_size, 1/sr)
    
    axs[0,0].plot(freqs[:N//2], 20*np.log10(np.abs(X_fft[:N//2]) + 1e-10), 'b-', linewidth=1.5, label='FFT')
    axs[0,0].set_title(f"{title} - Magnitude Spectrum (FFT)", fontweight='bold')
    axs[0,0].set_xlabel('Frequency (Hz)')
    axs[0,0].set_ylabel('Magnitude (dB)')
    axs[0,0].grid(True, alpha=0.3)
    axs[0,0].legend()
    
    axs[0,1].plot(freqs[:N//2], np.angle(X_fft[:N//2]), 'r-', linewidth=1.5)
    axs[0,1].set_title(f"{title} - Phase Spectrum", fontweight='bold')
    axs[0,1].set_xlabel('Frequency (Hz)')
    axs[0,1].set_ylabel('Phase (radians)')
    axs[0,1].grid(True, alpha=0.3)
    
    axs[1,0].plot(freqs_subset[:subset_size//2], np.abs(X_fft[:subset_size//2]), 'b-', label='FFT', alpha=0.7)
    axs[1,0].plot(freqs_subset[:subset_size//2], np.abs(X_dft_manual[:subset_size//2]), 'r--', label='Manual DFT', alpha=0.7)
    axs[1,0].set_title("DFT vs FFT Comparison", fontweight='bold')
    axs[1,0].set_xlabel('Frequency (Hz)')
    axs[1,0].set_ylabel('Magnitude')
    axs[1,0].legend()
    axs[1,0].grid(True, alpha=0.3)
    
    f, Pxx = signal.welch(audio, sr, nperseg=min(1024, N//4))
    axs[1,1].semilogy(f, Pxx, 'g-', linewidth=1.5)
    axs[1,1].set_title(f"{title} - Power Spectral Density", fontweight='bold')
    axs[1,1].set_xlabel('Frequency (Hz)')
    axs[1,1].set_ylabel('PSD (V²/Hz)')
    axs[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_convolution_analysis(x, h, title="Convolution Analysis"):
    """Plot convolution analysis"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    axs[0,0].stem(range(len(x)), x, basefmt=" ")
    axs[0,0].set_title("Input Signal x[n]", fontweight='bold')
    axs[0,0].grid(True, alpha=0.3)
    
    axs[0,1].stem(range(len(h)), h, basefmt=" ")
    axs[0,1].set_title("Impulse Response h[n]", fontweight='bold')
    axs[0,1].grid(True, alpha=0.3)
    
    y_linear = linear_convolution(x, h)
    axs[1,0].stem(range(len(y_linear)), y_linear, basefmt=" ")
    axs[1,0].set_title("Linear Convolution y[n] = x[n] * h[n]", fontweight='bold')
    axs[1,0].grid(True, alpha=0.3)
    
    N = max(len(x), len(h))
    x_padded = np.pad(x, (0, N - len(x)), 'constant')
    h_padded = np.pad(h, (0, N - len(h)), 'constant')
    y_circular = circular_convolution(x_padded, h_padded, N)
    axs[1,1].stem(range(len(y_circular)), y_circular, basefmt=" ")
    axs[1,1].set_title(f"Circular Convolution (N={N})", fontweight='bold')
    axs[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_mixed_vs_separated(mixed_signal, separated_sources, sr, title="Mixed vs Separated Signals"):
    """Plot mixed signal and separated sources in time and frequency domains"""
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    
    # Time Domain
    time = np.linspace(0, len(mixed_signal)/sr, len(mixed_signal))
    axs[0,0].plot(time, mixed_signal, label='Mixed Signal', color='blue')
    axs[0,0].set_title(f"{title} - Mixed Signal (Time)", fontweight='bold')
    axs[0,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Amplitude')
    axs[0,0].legend()
    axs[0,0].grid(True, alpha=0.3)
    
    axs[1,0].plot(time, separated_sources[0], label='Source 1 (FastICA)', color='green')
    axs[1,0].set_title(f"{title} - Separated Source 1 (Time)", fontweight='bold')
    axs[1,0].set_xlabel('Time (s)')
    axs[1,0].set_ylabel('Amplitude')
    axs[1,0].legend()
    axs[1,0].grid(True, alpha=0.3)
    
    axs[2,0].plot(time, separated_sources[1], label='Source 2 (FastICA)', color='red')
    axs[2,0].set_title(f"{title} - Separated Source 2 (Time)", fontweight='bold')
    axs[2,0].set_xlabel('Time (s)')
    axs[2,0].set_ylabel('Amplitude')
    axs[2,0].legend()
    axs[2,0].grid(True, alpha=0.3)
    
    # Frequency Domain
    freqs = fftfreq(len(mixed_signal), 1/sr)
    X_mixed = fft(mixed_signal)
    X_src1 = fft(separated_sources[0])
    X_src2 = fft(separated_sources[1])
    
    axs[0,1].plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(X_mixed[:len(X_mixed)//2]) + 1e-10), label='Mixed Signal', color='blue')
    axs[0,1].set_title(f"{title} - Mixed Signal (Frequency)", fontweight='bold')
    axs[0,1].set_xlabel('Frequency (Hz)')
    axs[0,1].set_ylabel('Magnitude (dB)')
    axs[0,1].legend()
    axs[0,1].grid(True, alpha=0.3)
    
    axs[1,1].plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(X_src1[:len(X_src1)//2]) + 1e-10), label='Source 1 (FastICA)', color='green')
    axs[1,1].set_title(f"{title} - Separated Source 1 (Frequency)", fontweight='bold')
    axs[1,1].set_xlabel('Frequency (Hz)')
    axs[1,1].set_ylabel('Magnitude (dB)')
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)
    
    axs[2,1].plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(X_src2[:len(X_src2)//2]) + 1e-10), label='Source 2 (FastICA)', color='red')
    axs[2,1].set_title(f"{title} - Separated Source 2 (Frequency)", fontweight='bold')
    axs[2,1].set_xlabel('Frequency (Hz)')
    axs[2,1].set_ylabel('Magnitude (dB)')
    axs[2,1].legend()
    axs[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_source_separation_analysis(mixed_signal, separated_sources, sr, method="FastICA"):
    """Plot comprehensive source separation analysis"""
    fig, axs = plt.subplots(5, 2, figsize=(16, 24))
    
    time = np.linspace(0, len(mixed_signal)/sr, len(mixed_signal))
    
    axs[0,0].plot(time, mixed_signal)
    axs[0,0].set_title("Mixed Signal - Time Domain", fontweight='bold')
    axs[0,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Amplitude')
    axs[0,0].grid(True, alpha=0.3)
    
    axs[0,1].set_visible(False)  # Only one mixed signal, hide second plot
    
    freqs = fftfreq(len(mixed_signal), 1/sr)
    X = fft(mixed_signal)
    
    axs[1,0].plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(X[:len(X)//2]) + 1e-10))
    axs[1,0].set_title("Mixed Signal - Frequency Domain", fontweight='bold')
    axs[1,0].set_xlabel('Frequency (Hz)')
    axs[1,0].set_ylabel('Magnitude (dB)')
    axs[1,0].grid(True, alpha=0.3)
    
    axs[1,1].set_visible(False)  # Hide second plot
    
    time_sep = np.linspace(0, len(separated_sources[0])/sr, len(separated_sources[0]))
    
    axs[2,0].plot(time_sep, separated_sources[0])
    axs[2,0].set_title(f"Separated Source 1 ({method}) - Time Domain", fontweight='bold')
    axs[2,0].set_xlabel('Time (s)')
    axs[2,0].set_ylabel('Amplitude')
    axs[2,0].grid(True, alpha=0.3)
    
    axs[2,1].plot(time_sep, separated_sources[1])
    axs[2,1].set_title(f"Separated Source 2 ({method}) - Time Domain", fontweight='bold')
    axs[2,1].set_xlabel('Time (s)')
    axs[2,1].set_ylabel('Amplitude')
    axs[2,1].grid(True, alpha=0.3)
    
    freqs_sep1 = fftfreq(len(separated_sources[0]), 1/sr)
    freqs_sep2 = fftfreq(len(separated_sources[1]), 1/sr)
    X_sep1 = fft(separated_sources[0])
    X_sep2 = fft(separated_sources[1])
    
    axs[3,0].plot(freqs_sep1[:len(freqs_sep1)//2], 20*np.log10(np.abs(X_sep1[:len(X_sep1)//2]) + 1e-10))
    axs[3,0].set_title(f"Separated Source 1 ({method}) - Frequency Domain", fontweight='bold')
    axs[3,0].set_xlabel('Frequency (Hz)')
    axs[3,0].set_ylabel('Magnitude (dB)')
    axs[3,0].grid(True, alpha=0.3)
    
    axs[3,1].plot(freqs_sep2[:len(freqs_sep2)//2], 20*np.log10(np.abs(X_sep2[:len(X_sep2)//2]) + 1e-10))
    axs[3,1].set_title(f"Separated Source 2 ({method}) - Frequency Domain", fontweight='bold')
    axs[3,1].set_xlabel('Frequency (Hz)')
    axs[3,1].set_ylabel('Magnitude (dB)')
    axs[3,1].grid(True, alpha=0.3)
    
    # Cross-correlation plot
    norm_factor = np.sqrt(np.sum(separated_sources[0]*2) * np.sum(separated_sources[1]*2)) + 1e-10
    cross_corr = signal.correlate(separated_sources[0], separated_sources[1], mode='full') / norm_factor
    lags = signal.correlation_lags(len(separated_sources[0]), len(separated_sources[1]), mode='full') / sr
    axs[4,0].plot(lags, cross_corr)
    axs[4,0].set_title(f"Cross-Correlation of Separated Sources ({method})", fontweight='bold')
    axs[4,0].set_xlabel('Lag (s)')
    axs[4,0].set_ylabel('Normalized Correlation')
    axs[4,0].grid(True, alpha=0.3)
    
    # Energy ratio visualization
    energy_ratio = compute_energy_ratio(separated_sources[0], separated_sources[1])
    axs[4,1].bar(['Source 1', 'Source 2'], [np.sum(separated_sources[0]*2), np.sum(separated_sources[1]*2)], color=['blue', 'red'])
    axs[4,1].set_title(f"Energy Comparison (Ratio: {energy_ratio:.2f})", fontweight='bold')
    axs[4,1].set_ylabel('Energy')
    
    plt.tight_layout()
    return fig

def plot_time_freq_domain(audio, sr, title="Time vs Frequency Domain"):
    """Plot time domain and frequency domain comparison"""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    time = np.linspace(0, len(audio)/sr, len(audio))
    axs[0].plot(time[:min(1000, len(audio))], audio[:min(1000, len(audio))])
    axs[0].set_title(f"{title} - Time Domain", fontweight='bold')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True, alpha=0.3)
    
    X = fft(audio)
    freqs = fftfreq(len(audio), 1/sr)
    axs[1].plot(freqs[:len(freqs)//2], np.abs(X[:len(X)//2]))
    axs[1].set_title(f"{title} - Frequency Domain", fontweight='bold')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def show_signals_systems_page():
    # Apply custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .upload-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        margin: 1rem 0;
        text-align: center;
    }
    .section-header {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .stMetric {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .stTabs {
        background: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #764ba2, #667eea);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    div[data-testid="metric-container"] > div {
        color: white;
    }
    .stAudio {
        margin: 1rem 0;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>📡 Advanced Signals & Systems Audio Analyzer</h1>
        <p>Comprehensive analysis featuring Source Separation (FastICA), Convolution, DFT/FFT, Spectrogram & Chroma, Time vs Frequency Domain, and Feature Extraction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with analysis options
    st.sidebar.markdown("## 🔬 Analysis Configuration")
    show_manual_dft = st.sidebar.checkbox("Show Manual DFT vs FFT", value=True)
    show_convolution = st.sidebar.checkbox("Show Convolution Analysis", value=True)
    show_spectrogram = st.sidebar.checkbox("Show Spectrogram Analysis", value=True)
    show_chroma = st.sidebar.checkbox("Show Chroma Features", value=True)
    show_source_separation = st.sidebar.checkbox("Perform Source Separation", value=True)
    show_feature_extraction = st.sidebar.checkbox("Extract Signal Features", value=True)
    
    model = load_model()
    
    # Input section
    st.sidebar.markdown("### 🎵 Input Method")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Audio File", "Record Audio"]
    )
    
    audio_data, sr = None, None
    separated_sources = None
    
    if input_method == "Upload Audio File":
        st.markdown('<div class="section-header">🎯 Upload Mixed Audio Signal</div>', unsafe_allow_html=True)
        st.markdown("Upload an audio signal (WAV, MP3, or FLAC) containing mixed sources for analysis and source separation.")
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### 🎧 Mixed Audio Signal")
        file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac"], key="mixed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if file:
            try:
                waveform, sr = torchaudio.load(file)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    st.warning("Converting audio from stereo to mono.")
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                audio_data = waveform.squeeze().numpy()
                st.success(f"✅ Successfully loaded audio file! Sample Rate: {sr} Hz, Duration: {len(audio_data)/sr:.2f}s")
                
            except Exception as e:
                st.error(f"Error loading audio file: {str(e)}")
                return
    
    elif input_method == "Record Audio":
        st.markdown('<div class="section-header">🎙 Record Audio Signal</div>', unsafe_allow_html=True)
        st.markdown("Record an audio signal containing mixed sources for analysis and source separation.")
        
        duration = st.slider("Recording duration (seconds)", 1, 10, 3)
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### 🎧 Record Mixed Audio Signal")
        if st.button("🎙 Record Signal", key="record"):
            with st.spinner("🔴 Recording audio..."):
                fs = 16000
                recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait()
                audio_data = recording.flatten()
                sr = fs
            st.success("✅ Recording complete!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if audio_data is not None:
        st.markdown('<div class="section-header">🎵 Uploaded/Recorded Signal</div>', unsafe_allow_html=True)
        st.markdown("*Mixed Signal*")
        temp_audio = tempfile.mktemp(suffix=".wav")
        sf.write(temp_audio, audio_data, sr)
        st.audio(temp_audio, format="audio/wav")
        
        st.markdown("---")
        
        # Create tabs for each analysis
        tabs = st.tabs([
            "Source Separation",
            "Convolution Analysis",
            "DFT/FFT Analysis",
            "Spectrogram & Chroma",
            "Time vs Frequency",
            "Feature Extraction"
        ])
        
        # Tab 1: Source Separation Analysis
        with tabs[0]:
            st.markdown('<div class="section-header">🛠 FastICA-based Blind Source Separation</div>', unsafe_allow_html=True)
            if show_source_separation:
                with st.spinner("🧠 Performing FastICA source separation..."):
                    try:
                        # Prepare single mono mixture
                        waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, length]
                        st.write(f"Input tensor shape: {waveform.shape}")
                        
                        if sr != 16000:
                            st.write(f"Resampling from {sr} Hz to 16000 Hz for model compatibility.")
                            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                            sr = 16000
                        
                        input_tensor = tensors_to_device(waveform, device="cpu")
                        model.to("cpu")
                        with torch.no_grad():
                            separated = model.separate(input_tensor)  # Output: [1, 2, length]
                        
                        st.write(f"Separated tensor shape: {separated.shape}")
                        
                        src1 = separated[0, 0].cpu().numpy()
                        src2 = separated[0, 1].cpu().numpy()
                        reduced_src1 = nr.reduce_noise(y=src1, sr=sr)
                        reduced_src2 = nr.reduce_noise(y=src2, sr=sr)
                        separated_sources = [reduced_src1, reduced_src2]
                        
                        sf.write("source1_clean.wav", reduced_src1, sr)
                        sf.write("source2_clean.wav", reduced_src2, sr)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("🗣 Separated Source 1 (FastICA)")
                            st.audio("source1_clean.wav", format="audio/wav")
                            if show_feature_extraction:
                                src1_features = compute_signal_features(reduced_src1, sr, ref_signal=audio_data)
                                st.write("*Source 1 Features:*")
                                for key, value in list(src1_features.items())[:7]:
                                    st.text(f"{key}: {value:.4f}")
                        with col2:
                            st.markdown("🗣 Separated Source 2 (FastICA)")
                            st.audio("source2_clean.wav", format="audio/wav")
                            if show_feature_extraction:
                                src2_features = compute_signal_features(reduced_src2, sr, ref_signal=audio_data)
                                st.write("*Source 2 Features:*")
                                for key, value in list(src2_features.items())[:7]:
                                    st.text(f"{key}: {value:.4f}")
                        
                        st.success("✅ Source separation completed successfully using FastICA!")
                        
                        # Download buttons for separated sources
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download Separated Source 1",
                                data=open("source1_clean.wav", "rb").read(),
                                file_name="separated_source_1.wav",
                                mime="audio/wav"
                            )
                        with col2:
                            st.download_button(
                                label="📥 Download Separated Source 2",
                                data=open("source2_clean.wav", "rb").read(),
                                file_name="separated_source_2.wav",
                                mime="audio/wav"
                            )
                        
                        # Mixed vs Separated Plot
                        st.markdown("#### Mixed vs Separated Signals")
                        fig_mixed_vs_sep = plot_mixed_vs_separated(audio_data, separated_sources, sr)
                        st.pyplot(fig_mixed_vs_sep)
                        
                        # Source Separation Visualization
                        st.markdown("#### Signal and Source Visualizations")
                        fig_sep = plot_source_separation_analysis(audio_data, separated_sources, sr, method="FastICA")
                        st.pyplot(fig_sep)
                        
                        # Energy Ratio
                        st.markdown("#### Energy Ratio Between Separated Sources")
                        energy_ratio = compute_energy_ratio(separated_sources[0], separated_sources[1])
                        st.metric("Energy Ratio (Source 1 / Source 2)", f"{energy_ratio:.2f}")
                    
                    except Exception as e:
                        st.error(f"Error during source separation: {str(e)}")
                        return
        
        # Tab 2: Convolution Analysis
        with tabs[1]:
            if show_convolution:
                st.markdown('<div class="section-header">🔄 Convolution Analysis</div>', unsafe_allow_html=True)
                h = np.array([0.5, 0.3, 0.2, 0.1])
                x_short = audio_data[:20]
                st.markdown("*Convolution with Sample Impulse Response (Mixed Signal)*")
                fig = plot_convolution_analysis(x_short, h, "Convolution Demo (Mixed Signal)")
                st.pyplot(fig)
                
                st.markdown("#### 🎯 Convolution Application")
                filtered_linear = np.convolve(audio_data, h, mode='same')
                temp_filtered = tempfile.mktemp(suffix=".wav")
                sf.write(temp_filtered, filtered_linear, sr)
                st.markdown("*Mixed Signal After Convolution*")
                st.audio(temp_filtered, format="audio/wav")
                
                if show_source_separation and separated_sources is not None:
                    st.markdown("*Convolution for Separated Sources*")
                    col1, col2 = st.columns(2)
                    with col1:
                        filtered_src1 = np.convolve(separated_sources[0], h, mode='same')
                        temp_filtered_src1 = tempfile.mktemp(suffix=".wav")
                        sf.write(temp_filtered_src1, filtered_src1, sr)
                        st.markdown("*Separated Source 1 After Convolution*")
                        st.audio(temp_filtered_src1, format="audio/wav")
                    with col2:
                        filtered_src2 = np.convolve(separated_sources[1], h, mode='same')
                        temp_filtered_src2 = tempfile.mktemp(suffix=".wav")
                        sf.write(temp_filtered_src2, filtered_src2, sr)
                        st.markdown("*Separated Source 2 After Convolution*")
                        st.audio(temp_filtered_src2, format="audio/wav")
        
        # Tab 3: DFT/FFT Analysis
        with tabs[2]:
            if show_manual_dft:
                st.markdown('<div class="section-header">🔍 DFT/FFT Analysis</div>', unsafe_allow_html=True)
                st.markdown("*Mixed Signal Analysis*")
                fig = plot_dft_analysis(audio_data, sr, "Mixed Signal")
                st.pyplot(fig)
                
                if show_source_separation and separated_sources is not None:
                    st.markdown("*Separated Source 1 Analysis*")
                    fig_src1 = plot_dft_analysis(separated_sources[0], sr, "Separated Source 1 (FastICA)")
                    st.pyplot(fig_src1)
                    
                    st.markdown("*Separated Source 2 Analysis*")
                    fig_src2 = plot_dft_analysis(separated_sources[1], sr, "Separated Source 2 (FastICA)")
                    st.pyplot(fig_src2)
        
        # Tab 4: Spectrogram & Chroma Analysis
        with tabs[3]:
            if show_spectrogram or show_chroma:
                st.markdown('<div class="section-header">🎵 Spectrogram & Chroma Analysis</div>', unsafe_allow_html=True)
                st.markdown("*Mixed Signal Analysis*")
                fig = plot_spectrogram_and_chroma(audio_data, sr, "Mixed Signal")
                st.pyplot(fig)
                
                if show_chroma:
                    st.markdown("*Mixed Signal Chroma Features*")
                    chroma_features, _ = compute_chroma_features(audio_data, sr)
                    if chroma_features:
                        chroma_df = pd.DataFrame(list(chroma_features.items()), columns=['Chroma Feature', 'Value'])
                        chroma_df['Value'] = chroma_df['Value'].apply(lambda x: f"{x:.6f}")
                        st.dataframe(chroma_df, use_container_width=True)
                
                if show_source_separation and separated_sources is not None:
                    st.markdown("*Separated Source 1 Analysis*")
                    fig_src1 = plot_spectrogram_and_chroma(separated_sources[0], sr, "Separated Source 1 (FastICA)")
                    st.pyplot(fig_src1)
                    
                    if show_chroma:
                        st.markdown("*Separated Source 1 Chroma Features*")
                        chroma_features_src1, _ = compute_chroma_features(separated_sources[0], sr)
                        if chroma_features_src1:
                            chroma_df_src1 = pd.DataFrame(list(chroma_features_src1.items()), columns=['Chroma Feature', 'Value'])
                            chroma_df_src1['Value'] = chroma_df_src1['Value'].apply(lambda x: f"{x:.6f}")
                            st.dataframe(chroma_df_src1, use_container_width=True)
                    
                    st.markdown("*Separated Source 2 Analysis*")
                    fig_src2 = plot_spectrogram_and_chroma(separated_sources[1], sr, "Separated Source 2 (FastICA)")
                    st.pyplot(fig_src2)
                    
                    if show_chroma:
                        st.markdown("*Separated Source 2 Chroma Features*")
                        chroma_features_src2, _ = compute_chroma_features(separated_sources[1], sr)
                        if chroma_features_src2:
                            chroma_df_src2 = pd.DataFrame(list(chroma_features_src2.items()), columns=['Chroma Feature', 'Value'])
                            chroma_df_src2['Value'] = chroma_df_src2['Value'].apply(lambda x: f"{x:.6f}")
                            st.dataframe(chroma_df_src2, use_container_width=True)
        
        # Tab 5: Time vs Frequency Domain
        with tabs[4]:
            st.markdown('<div class="section-header">📊 Time vs Frequency Domain</div>', unsafe_allow_html=True)
            st.markdown("*Mixed Signal*")
            fig = plot_time_freq_domain(audio_data, sr, "Mixed Signal")
            st.pyplot(fig)
            
            if show_source_separation and separated_sources is not None:
                st.markdown("*Separated Source 1 (FastICA)*")
                fig_src1 = plot_time_freq_domain(separated_sources[0], sr, "Separated Source 1 (FastICA)")
                st.pyplot(fig_src1)
                
                st.markdown("*Separated Source 2 (FastICA)*")
                fig_src2 = plot_time_freq_domain(separated_sources[1], sr, "Separated Source 2 (FastICA)")
                st.pyplot(fig_src2)
        
        # Tab 6: Feature Extraction
        with tabs[5]:
            if show_feature_extraction:
                st.markdown('<div class="section-header">🎯 Signal Feature Extraction</div>', unsafe_allow_html=True)
                features = compute_signal_features(audio_data, sr)
                
                st.markdown("*Mixed Signal Features*")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("📊 Time Domain Features**")
                    st.metric("Mean", f"{features['Mean']:.6f}")
                    st.metric("RMS", f"{features['RMS']:.4f}")
                    st.metric("Peak", f"{features['Peak']:.4f}")
                    st.metric("Crest Factor", f"{features['Crest Factor']:.2f}")
                with col2:
                    st.markdown("🎵 Frequency Domain Features**")
                    st.metric("Spectral Centroid", f"{features['Spectral Centroid']:.0f} Hz")
                    st.metric("Spectral Bandwidth", f"{features['Spectral Bandwidth']:.0f} Hz")
                    st.metric("Spectral Rolloff", f"{features['Spectral Rolloff']:.0f} Hz")
                with col3:
                    st.markdown("📈 Statistical Features**")
                    st.metric("Variance", f"{features['Variance']:.6f}")
                    st.metric("Skewness", f"{features['Skewness']:.3f}")
                    st.metric("Kurtosis", f"{features['Kurtosis']:.3f}")
                    st.metric("SNR (dB)", f"{features['SNR_dB']:.2f}")
                
                st.markdown("#### 📋 Complete Feature Table (Mixed Signal)")
                feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                feature_df['Value'] = feature_df['Value'].apply(lambda x: f"{x:.6f}")
                st.dataframe(feature_df, use_container_width=True)
                
                if show_source_separation and separated_sources is not None:
                    st.markdown("*Separated Source 1 Features (FastICA)*")
                    src1_features = compute_signal_features(separated_sources[0], sr, ref_signal=audio_data)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("📊 Time Domain Features**")
                        st.metric("Mean", f"{src1_features['Mean']:.6f}")
                        st.metric("RMS", f"{src1_features['RMS']:.4f}")
                        st.metric("Peak", f"{src1_features['Peak']:.4f}")
                        st.metric("Crest Factor", f"{src1_features['Crest Factor']:.2f}")
                    with col2:
                        st.markdown("🎵 Frequency Domain Features**")
                        st.metric("Spectral Centroid", f"{src1_features['Spectral Centroid']:.0f} Hz")
                        st.metric("Spectral Bandwidth", f"{src1_features['Spectral Bandwidth']:.0f} Hz")
                        st.metric("Spectral Rolloff", f"{src1_features['Spectral Rolloff']:.0f} Hz")
                    with col3:
                        st.markdown("📈 Statistical Features**")
                        st.metric("Variance", f"{src1_features['Variance']:.6f}")
                        st.metric("Skewness", f"{src1_features['Skewness']:.3f}")
                        st.metric("Kurtosis", f"{src1_features['Kurtosis']:.3f}")
                        st.metric("SNR (dB)", f"{src1_features['SNR_dB']:.2f}")
                        st.metric("SDR (dB)", f"{src1_features['SDR_dB']:.2f}")
                        st.metric("Cross-Correlation Peak", f"{src1_features['Cross_Correlation_Peak']:.3f}")
                    
                    st.markdown("#### 📋 Complete Feature Table (Source 1)")
                    src1_df = pd.DataFrame(list(src1_features.items()), columns=['Feature', 'Value'])
                    src1_df['Value'] = src1_df['Value'].apply(lambda x: f"{x:.6f}")
                    st.dataframe(src1_df, use_container_width=True)
                    
                    st.markdown("*Separated Source 2 Features (FastICA)*")
                    src2_features = compute_signal_features(separated_sources[1], sr, ref_signal=audio_data)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("📊 Time Domain Features**")
                        st.metric("Mean", f"{src2_features['Mean']:.6f}")
                        st.metric("RMS", f"{src2_features['RMS']:.4f}")
                        st.metric("Peak", f"{src2_features['Peak']:.4f}")
                        st.metric("Crest Factor", f"{src2_features['Crest Factor']:.2f}")
                    with col2:
                        st.markdown("🎵 Frequency Domain Features**")
                        st.metric("Spectral Centroid", f"{src2_features['Spectral Centroid']:.0f} Hz")
                        st.metric("Spectral Bandwidth", f"{src2_features['Spectral Bandwidth']:.0f} Hz")
                        st.metric("Spectral Rolloff", f"{src2_features['Spectral Rolloff']:.0f} Hz")
                    with col3:
                        st.markdown("📈 Statistical Features**")
                        st.metric("Variance", f"{src2_features['Variance']:.6f}")
                        st.metric("Skewness", f"{src2_features['Skewness']:.3f}")
                        st.metric("Kurtosis", f"{src2_features['Kurtosis']:.3f}")
                        st.metric("SNR (dB)", f"{src2_features['SNR_dB']:.2f}")
                        st.metric("SDR (dB)", f"{src2_features['SDR_dB']:.2f}")
                        st.metric("Cross-Correlation Peak", f"{src2_features['Cross_Correlation_Peak']:.3f}")
                    
                    st.markdown("#### 📋 Complete Feature Table (Source 2)")
                    src2_df = pd.DataFrame(list(src2_features.items()), columns=['Feature', 'Value'])
                    src2_df['Value'] = src2_df['Value'].apply(lambda x: f"{x:.6f}")
                    st.dataframe(src2_df, use_container_width=True)
        
        # Analysis Report
        st.markdown("---")
        st.markdown("### 💾 Download Analysis Report")
        analysis_report = f"""
# Signals & Systems Analysis Report
Generated on: {pd.Timestamp.now()}

## Signal Properties
- Sample Rate: {sr} Hz
- Duration: {len(audio_data)/sr:.2f} seconds
- Total Samples: {len(audio_data)}

## Source Separation (FastICA)
- Separated into 2 sources using FastICA
"""
        if show_source_separation and separated_sources is not None:
            src1_features = compute_signal_features(separated_sources[0], sr, ref_signal=audio_data)
            src2_features = compute_signal_features(separated_sources[1], sr, ref_signal=audio_data)
            analysis_report += f"""
- Energy Ratio (Source 1 / Source 2): {compute_energy_ratio(separated_sources[0], separated_sources[1]):.2f}
"""
            analysis_report += f"""
### Feature Analysis (Separated Source 1)
"""
            for feature, value in src1_features.items():
                analysis_report += f"- {feature}: {value:.6f}\n"
            analysis_report += f"""
### Feature Analysis (Separated Source 2)
"""
            for feature, value in src2_features.items():
                analysis_report += f"- {feature}: {value:.6f}\n"
        
        analysis_report += f"""
## Feature Analysis (Mixed Signal)
"""
        if show_feature_extraction:
            features = compute_signal_features(audio_data, sr)
            for feature, value in features.items():
                analysis_report += f"- {feature}: {value:.6f}\n"
        
        analysis_report += f"""
## Chroma Analysis (Mixed Signal)
"""
        if show_chroma:
            chroma_features, _ = compute_chroma_features(audio_data, sr)
            for feature, value in chroma_features.items():
                analysis_report += f"- {feature}: {value:.6f}\n"
        
        if show_source_separation and separated_sources is not None:
            analysis_report += f"""
## Chroma Analysis (Separated Source 1)
"""
            if show_chroma:
                chroma_features_src1, _ = compute_chroma_features(separated_sources[0], sr)
                for feature, value in chroma_features_src1.items():
                    analysis_report += f"- {feature}: {value:.6f}\n"
            
            analysis_report += f"""
## Chroma Analysis (Separated Source 2)
"""
            if show_chroma:
                chroma_features_src2, _ = compute_chroma_features(separated_sources[1], sr)
                for feature, value in chroma_features_src2.items():
                    analysis_report += f"- {feature}: {value:.6f}\n"
        
        analysis_report += f"""
## Analysis Performed
"""
        if show_source_separation:
            analysis_report += "- Source Separation Analysis (FastICA)\n"
        if show_convolution:
            analysis_report += "- Convolution Analysis\n"
        if show_manual_dft:
            analysis_report += "- DFT/FFT Analysis\n"
        if show_spectrogram or show_chroma:
            analysis_report += "- Spectrogram & Chroma Analysis\n"
        analysis_report += "- Time vs Frequency Domain Analysis\n"
        if show_feature_extraction:
            analysis_report += "- Feature Extraction\n"
        
        analysis_report += f"""
## Theoretical Background
### FastICA Source Separation
FastICA is an efficient algorithm for Independent Component Analysis, used to separate a mixed signal into independent sources, effective for tasks like the cocktail party problem.

### Discrete Fourier Transform (DFT)
The DFT of a sequence x[n] of length N is defined as:
X[k] = Σ(n=0 to N-1) x[n] * e^(-j2πkn/N)

### Linear Convolution
y[n] = x[n] * h[n] = Σ(k=-∞ to ∞) x[k] * h[n-k]

### Spectrogram
A time-frequency representation showing how the spectral content of a signal evolves over time.

### Chroma Features
Projection of the spectrum onto 12 pitch classes, capturing harmonic and melodic characteristics.
"""
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=analysis_report,
            file_name="signals_systems_analysis_report.txt",
            mime="text/plain"
        )
        
        # Concepts Summary
        st.markdown("### 🎓 Signals & Systems Concepts Summary")
        concept_summary = {
            "FastICA Source Separation": "An algorithm for separating a mixed signal into independent sources, solving problems like the cocktail party effect.",
            "Convolution": "Mathematical operation fundamental to LTI systems. Linear convolution for infinite sequences, circular for finite periodic sequences.",
            "DFT/FFT": "Discrete Fourier Transform converts time-domain signals to frequency domain. FFT is an efficient algorithm for computing DFT.",
            "Spectrogram Analysis": "Visual representation of the spectrum of frequencies of a signal as it varies with time.",
            "Chroma Features": "Capture the harmonic content of an audio signal by projecting it onto 12 pitch classes.",
            "Feature Extraction": "Process of extracting meaningful characteristics from signals for analysis and classification, including SNR and SDR."
        }
        
        for concept, description in concept_summary.items():
            with st.expander(f"📚 {concept}"):
                st.write(description)

if __name__ == "_main_":
    show_signals_systems_page()


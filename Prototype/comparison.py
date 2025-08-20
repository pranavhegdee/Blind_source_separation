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
from sklearn.decomposition import FastICA, PCA, NMF
import tempfile
import time
import pandas as pd

@st.cache_resource
def load_comparison_models():
    """Load available BSS models for comparison"""
    models = {}
    try:
        models['ConvTasNet_Libri2Mix'] = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
        models['ConvTasNet_Basic'] = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
        for model in models.values():
            model.eval()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        models['ConvTasNet_Basic'] = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
        models['ConvTasNet_Basic'].eval()
    return models

def apply_fastica(audio, sr, n_components=2):
    """Apply FastICA for blind source separation on single-channel audio"""
    try:
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Create pseudo-multichannel signal using time-shifted versions
        shift_samples = int(sr * 0.01)  # 10ms shift
        X = np.vstack([audio, np.roll(audio, shift_samples)]).T  # Shape: (n_samples, 2)
        
        # Apply FastICA
        ica = FastICA(n_components=n_components, random_state=42)
        S_ = ica.fit_transform(X)  # Shape: (n_samples, n_components)
        
        # Extract sources
        sources = [S_[:, i] for i in range(n_components)]
        
        # Ensure sources match original audio length
        sources = [s[:len(audio)] for s in sources]
        
        # Pad with zeros if fewer than 2 sources
        while len(sources) < 2:
            sources.append(np.zeros_like(audio))
        
        return sources
    except Exception as e:
        st.error(f"FastICA processing failed: {e}")
        return [audio, np.zeros_like(audio)]

def apply_pca(audio, sr, n_components=2):
    """Apply PCA for blind source separation on single-channel audio"""
    try:
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Create pseudo-multichannel signal using time-shifted versions
        shift_samples = int(sr * 0.01)  # 10ms shift
        X = np.vstack([audio, np.roll(audio, shift_samples)]).T  # Shape: (n_samples, 2)
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        S_ = pca.fit_transform(X)  # Shape: (n_samples, n_components)
        
        # Extract sources
        sources = [S_[:, i] for i in range(n_components)]
        
        # Ensure sources match original audio length
        sources = [s[:len(audio)] for s in sources]
        
        # Pad with zeros if fewer than 2 sources
        while len(sources) < 2:
            sources.append(np.zeros_like(audio))
        
        return sources
    except Exception as e:
        st.error(f"PCA processing failed: {e}")
        return [audio, np.zeros_like(audio)]

def apply_nmf(audio, sr, n_components=2):
    """Apply NMF for blind source separation on single-channel audio"""
    try:
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Create magnitude spectrogram for NMF (non-negative input required)
        window = np.hanning(512)
        hop_length = 128
        X = np.abs(np.array([np.fft.fft(audio[i:i+512] * window) for i in range(0, len(audio)-512, hop_length)]))
        
        # Apply NMF
        nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
        W = nmf.fit_transform(X)  # Shape: (n_frames, n_components)
        H = nmf.components_  # Shape: (n_components, n_features)
        
        # Reconstruct sources
        sources = []
        for i in range(n_components):
            # Reconstruct using original phase approximation
            component = np.outer(W[:, i], H[i, :])
            reconstructed = np.zeros_like(audio)
            for j in range(len(component)):
                start = j * hop_length
                end = start + 512
                if end <= len(audio):
                    reconstructed[start:end] += component[j, :512] * np.sign(audio[start:end])
            sources.append(reconstructed[:len(audio)])
        
        # Pad with zeros if fewer than 2 sources
        while len(sources) < 2:
            sources.append(np.zeros_like(audio))
        
        return sources
    except Exception as e:
        st.error(f"NMF processing failed: {e}")
        return [audio, np.zeros_like(audio)]

def calculate_si_sdr(reference, estimated):
    """Calculate Scale-Invariant Signal-to-Distortion Ratio"""
    try:
        alpha = np.dot(estimated, reference) / np.dot(reference, reference)
        sdr = 10 * np.log10(np.sum((alpha * reference) ** 2) / np.sum((alpha * reference - estimated) ** 2))
        return sdr if np.isfinite(sdr) else -np.inf
    except:
        return -np.inf

def calculate_pesq_approx(reference, estimated, sr):
    """Approximate PESQ calculation using correlation"""
    try:
        ref_norm = (reference - np.mean(reference)) / np.std(reference)
        est_norm = (estimated - np.mean(estimated)) / np.std(estimated)
        correlation = np.corrcoef(ref_norm, est_norm)[0, 1]
        pesq_approx = 1 + 4 * max(0, correlation)
        return pesq_approx
    except:
        return 1.0

def plot_comparison_waveforms(sources_dict, sr, title_prefix=""):
    """Plot waveforms for different models"""
    fig, axes = plt.subplots(len(sources_dict), 2, figsize=(15, 3*len(sources_dict)))
    
    if len(sources_dict) == 1:
        axes = np.array([axes])
    
    colors = ['#667eea', '#f093fb', '#4ecdc4', '#45b7d1']
    
    for idx, (model_name, (src1, src2)) in enumerate(sources_dict.items()):
        color = colors[idx % len(colors)]
        axes[idx, 0].plot(src1, color=color, linewidth=1.5, alpha=0.8)
        axes[idx, 0].set_title(f"{title_prefix}{model_name} - Source 1", fontweight='bold')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].set_ylabel('Amplitude')
        
        axes[idx, 1].plot(src2, color=color, linewidth=1.5, alpha=0.8)
        axes[idx, 1].set_title(f"{title_prefix}{model_name} - Source 2", fontweight='bold')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_ylabel('Amplitude')
    
    axes[-1, 0].set_xlabel('Time (samples)')
    axes[-1, 1].set_xlabel('Time (samples)')
    
    plt.tight_layout()
    st.pyplot(fig)

def create_metrics_dataframe(metrics_dict):
    """Create a pandas DataFrame from metrics dictionary"""
    df_data = []
    for model_name, metrics in metrics_dict.items():
        df_data.append({
            'Model': model_name,
            'SI-SDR Source 1 (dB)': f"{metrics['si_sdr_src1']:.2f}",
            'SI-SDR Source 2 (dB)': f"{metrics['si_sdr_src2']:.2f}",
            'Average SI-SDR (dB)': f"{metrics['avg_si_sdr']:.2f}",
            'PESQ Source 1': f"{metrics['pesq_src1']:.2f}",
            'PESQ Source 2': f"{metrics['pesq_src2']:.2f}",
            'Processing Time (s)': f"{metrics['processing_time']:.3f}",
            'Model Size': metrics['model_size']
        })
    return pd.DataFrame(df_data)

def main():
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ BSS Model Comparison</h1>
        <p>Compare different Blind Source Separation techniques and models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "📁 Upload Audio File"
    if 'waveform' not in st.session_state:
        st.session_state.waveform = None
    if 'sr' not in st.session_state:
        st.session_state.sr = None
    if 'audio_processed' not in st.session_state:
        st.session_state.audio_processed = False
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {
            'separated_sources': {},
            'processing_times': {},
            'model_metrics': {},
            'selected_methods': []
        }
    
    # Load neural network models
    with st.spinner("🔄 Loading comparison models..."):
        models = load_comparison_models()
    
    st.success(f"✅ Loaded {len(models)} neural network models for comparison")
    
    # Model information
    with st.expander("📋 Model Information"):
        model_info = {
            'ConvTasNet_Libri2Mix': 'Trained on LibriMix dataset, optimized for clean speech separation',
            'ConvTasNet_Basic': 'Basic ConvTasNet model (fallback option)',
            'FastICA': 'Statistical method using Independent Component Analysis, computationally efficient',
            'PCA': 'Principal Component Analysis, reduces dimensionality to separate sources',
            'NMF': 'Non-negative Matrix Factorization, suitable for non-negative signal decomposition'
        }
        for model_name in list(models.keys()) + ['FastICA', 'PCA', 'NMF']:
            st.write(f"**{model_name}**: {model_info.get(model_name, 'No description available')}")
    
    st.markdown("### 🎯 Select Your Input Method:")
    input_method = st.radio(
        "Choose how you want to provide audio:",
        ["📁 Upload Audio File", "🎤 Record via Microphone"],
        key="input_method_radio",
        help="Choose between uploading a pre-recorded file or recording live audio"
    )
    
    # Update session state
    if input_method != st.session_state.input_method:
        st.session_state.input_method = input_method
        st.session_state.waveform = None
        st.session_state.sr = None
        st.session_state.audio_processed = False
        st.session_state.comparison_results = {
            'separated_sources': {},
            'processing_times': {},
            'model_metrics': {},
            'selected_methods': []
        }
    
    if not st.session_state.audio_processed:
        if input_method == "📁 Upload Audio File":
            st.markdown("#### 📂 File Upload")
            uploaded_file = st.file_uploader(
                "Upload a 2-speaker mixed WAV file",
                type=["wav"],
                help="Upload a WAV file containing mixed audio from multiple speakers"
            )
            if uploaded_file is not None:
                st.session_state.waveform, st.session_state.sr = torchaudio.load(uploaded_file)
                st.session_state.audio_processed = True
                st.session_state.comparison_results = {
                    'separated_sources': {},
                    'processing_times': {},
                    'model_metrics': {},
                    'selected_methods': []
                }
                st.success("✅ File uploaded successfully!")
                
        elif input_method == "🎤 Record via Microphone":
            st.markdown("#### 🎙️ Live Recording")
            duration = st.slider("Recording duration (seconds)", 1, 20, 5)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🎙️ Start Recording", type="primary"):
                    with st.spinner("🔴 Recording in progress..."):
                        fs = 16000
                        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                        sd.wait()
                        temp_path =agens('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k')
        models['ConvTasNet_Basic'].eval()
    return models

def apply_fastica(audio, sr, n_components=2):
    """Apply FastICA for blind source separation on single-channel audio"""
    try:
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Create pseudo-multichannel signal using time-shifted versions
        shift_samples = int(sr * 0.01)  # 10ms shift
        X = np.vstack([audio, np.roll(audio, shift_samples)]).T  # Shape: (n_samples, 2)
        
        # Apply FastICA
        ica = FastICA(n_components=n_components, random_state=42)
        S_ = ica.fit_transform(X)  # Shape: (n_samples, n_components)
        
        # Extract sources
        sources = [S_[:, i] for i in range(n_components)]
        
        # Ensure sources match original audio length
        sources = [s[:len(audio)] for s in sources]
        
        # Pad with zeros if fewer than 2 sources
        while len(sources) < 2:
            sources.append(np.zeros_like(audio))
        
        return sources
    except Exception as e:
        st.error(f"FastICA processing failed: {e}")
        return [audio, np.zeros_like(audio)]

def apply_pca(audio, sr, n_components=2):
    """Apply PCA for blind source separation on single-channel audio"""
    try:
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Create pseudo-multichannel signal using time-shifted versions
        shift_samples = int(sr * 0.01)  # 10ms shift
        X = np.vstack([audio, np.roll(audio, shift_samples)]).T  # Shape: (n_samples, 2)
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        S_ = pca.fit_transform(X)  # Shape: (n_samples, n_components)
        
        # Extract sources
        sources = [S_[:, i] for i in range(n_components)]
        
        # Ensure sources match original audio length
        sources = [s[:len(audio)] for s in sources]
        
        # Pad with zeros if fewer than 2 sources
        while len(sources) < 2:
            sources.append(np.zeros_like(audio))
        
        return sources
    except Exception as e:
        st.error(f"PCA processing failed: {e}")
        return [audio, np.zeros_like(audio)]

def apply_nmf(audio, sr, n_components=2):
    """Apply NMF for blind source separation on single-channel audio"""
    try:
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Create magnitude spectrogram for NMF (non-negative input required)
        window = np.hanning(512)
        hop_length = 128
        X = np.abs(np.array([np.fft.fft(audio[i:i+512] * window) for i in range(0, len(audio)-512, hop_length)]))
        
        # Apply NMF
        nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
        W = nmf.fit_transform(X)  # Shape: (n_frames, n_components)
        H = nmf.components_  # Shape: (n_components, n_features)
        
        # Reconstruct sources
        sources = []
        for i in range(n_components):
            # Reconstruct using original phase approximation
            component = np.outer(W[:, i], H[i, :])
            reconstructed = np.zeros_like(audio)
            for j in range(len(component)):
                start = j * hop_length
                end = start + 512
                if end <= len(audio):
                    reconstructed[start:end] += component[j, :512] * np.sign(audio[start:end])
            sources.append(reconstructed[:len(audio)])
        
        # Pad with zeros if fewer than 2 sources
        while len(sources) < 2:
            sources.append(np.zeros_like(audio))
        
        return sources
    except Exception as e:
        st.error(f"NMF processing failed: {e}")
        return [audio, np.zeros_like(audio)]

def calculate_si_sdr(reference, estimated):
    """Calculate Scale-Invariant Signal-to-Distortion Ratio"""
    try:
        alpha = np.dot(estimated, reference) / np.dot(reference, reference)
        sdr = 10 * np.log10(np.sum((alpha * reference) ** 2) / np.sum((alpha * reference - estimated) ** 2))
        return sdr if np.isfinite(sdr) else -np.inf
    except:
        return -np.inf

def calculate_pesq_approx(reference, estimated, sr):
    """Approximate PESQ calculation using correlation"""
    try:
        ref_norm = (reference - np.mean(reference)) / np.std(reference)
        est_norm = (estimated - np.mean(estimated)) / np.std(estimated)
        correlation = np.corrcoef(ref_norm, est_norm)[0, 1]
        pesq_approx = 1 + 4 * max(0, correlation)
        return pesq_approx
    except:
        return 1.0

def plot_comparison_waveforms(sources_dict, sr, title_prefix=""):
    """Plot waveforms for different models"""
    fig, axes = plt.subplots(len(sources_dict), 2, figsize=(15, 3*len(sources_dict)))
    
    if len(sources_dict) == 1:
        axes = np.array([axes])
    
    colors = ['#667eea', '#f093fb', '#4ecdc4', '#45b7d1']
    
    for idx, (model_name, (src1, src2)) in enumerate(sources_dict.items()):
        color = colors[idx % len(colors)]
        axes[idx, 0].plot(src1, color=color, linewidth=1.5, alpha=0.8)
        axes[idx, 0].set_title(f"{title_prefix}{model_name} - Source 1", fontweight='bold')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].set_ylabel('Amplitude')
        
        axes[idx, 1].plot(src2, color=color, linewidth=1.5, alpha=0.8)
        axes[idx, 1].set_title(f"{title_prefix}{model_name} - Source 2", fontweight='bold')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_ylabel('Amplitude')
    
    axes[-1, 0].set_xlabel('Time (samples)')
    axes[-1, 1].set_xlabel('Time (samples)')
    
    plt.tight_layout()
    st.pyplot(fig)

def create_metrics_dataframe(metrics_dict):
    """Create a pandas DataFrame from metrics dictionary"""
    df_data = []
    for model_name, metrics in metrics_dict.items():
        df_data.append({
            'Model': model_name,
            'SI-SDR Source 1 (dB)': f"{metrics['si_sdr_src1']:.2f}",
            'SI-SDR Source 2 (dB)': f"{metrics['si_sdr_src2']:.2f}",
            'Average SI-SDR (dB)': f"{metrics['avg_si_sdr']:.2f}",
            'PESQ Source 1': f"{metrics['pesq_src1']:.2f}",
            'PESQ Source 2': f"{metrics['pesq_src2']:.2f}",
            'Processing Time (s)': f"{metrics['processing_time']:.3f}",
            'Model Size': metrics['model_size']
        })
    return pd.DataFrame(df_data)

def show_comparison_page():
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ BSS Model Comparison</h1>
        <p>Compare different Blind Source Separation techniques and models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "📁 Upload Audio File"
    if 'waveform' not in st.session_state:
        st.session_state.waveform = None
    if 'sr' not in st.session_state:
        st.session_state.sr = None
    if 'audio_processed' not in st.session_state:
        st.session_state.audio_processed = False
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {
            'separated_sources': {},
            'processing_times': {},
            'model_metrics': {},
            'selected_methods': []
        }
    
    # Load neural network models
    with st.spinner("🔄 Loading comparison models..."):
        models = load_comparison_models()
    
    st.success(f"✅ Loaded {len(models)} neural network models for comparison")
    
    # Model information
    with st.expander("📋 Model Information"):
        model_info = {
            'ConvTasNet_Libri2Mix': 'Trained on LibriMix dataset, optimized for clean speech separation',
            'ConvTasNet_Basic': 'Basic ConvTasNet model (fallback option)',
            'FastICA': 'Statistical method using Independent Component Analysis, computationally efficient',
            'PCA': 'Principal Component Analysis, reduces dimensionality to separate sources',
            'NMF': 'Non-negative Matrix Factorization, suitable for non-negative signal decomposition'
        }
        for model_name in list(models.keys()) + ['FastICA', 'PCA', 'NMF']:
            st.write(f"**{model_name}**: {model_info.get(model_name, 'No description available')}")
    
    st.markdown("### 🎯 Select Your Input Method:")
    input_method = st.radio(
        "Choose how you want to provide audio:",
        ["📁 Upload Audio File", "🎤 Record via Microphone"],
        key="input_method_radio",
        help="Choose between uploading a pre-recorded file or recording live audio"
    )
    
    # Update session state
    if input_method != st.session_state.input_method:
        st.session_state.input_method = input_method
        st.session_state.waveform = None
        st.session_state.sr = None
        st.session_state.audio_processed = False
        st.session_state.comparison_results = {
            'separated_sources': {},
            'processing_times': {},
            'model_metrics': {},
            'selected_methods': []
        }
    
    if not st.session_state.audio_processed:
        if input_method == "📁 Upload Audio File":
            st.markdown("#### 📂 File Upload")
            uploaded_file = st.file_uploader(
                "Upload a 2-speaker mixed WAV file",
                type=["wav"],
                help="Upload a WAV file containing mixed audio from multiple speakers"
            )
            if uploaded_file is not None:
                st.session_state.waveform, st.session_state.sr = torchaudio.load(uploaded_file)
                st.session_state.audio_processed = True
                st.session_state.comparison_results = {
                    'separated_sources': {},
                    'processing_times': {},
                    'model_metrics': {},
                    'selected_methods': []
                }
                st.success("✅ File uploaded successfully!")
                
        elif input_method == "🎤 Record via Microphone":
            st.markdown("#### 🎙️ Live Recording")
            duration = st.slider("Recording duration (seconds)", 1, 20, 5)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🎙️ Start Recording", type="primary"):
                    with st.spinner("🔴 Recording in progress..."):
                        fs = 16000
                        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                        sd.wait()
                        temp_path = tempfile.mktemp(suffix=".wav")
                        wavio.write(temp_path, recording, fs, sampwidth=2)
                        st.session_state.waveform, st.session_state.sr = torchaudio.load(temp_path)
                        st.session_state.audio_processed = True
                        st.session_state.comparison_results = {
                            'separated_sources': {},
                            'processing_times': {},
                            'model_metrics': {},
                            'selected_methods': []
                        }
                    st.success("✅ Recording complete!")
                    st.audio(temp_path, format="audio/wav")
    
    if st.session_state.waveform is not None:
        waveform = st.session_state.waveform
        sr = st.session_state.sr
        
        # Preprocessing
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            st.warning(f"⚠️ Resampling from {sr} Hz to 16kHz...")
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000
        st.session_state.waveform = waveform
        st.session_state.sr = sr
        
        # Display original audio
        st.markdown("---")
        st.subheader("🎧 Input Mixture")
        temp_mix = tempfile.mktemp(suffix=".wav")
        sf.write(temp_mix, waveform.squeeze().numpy(), sr)
        st.audio(temp_mix, format="audio/wav")
        
        # Comparison settings
        st.markdown("---")
        st.subheader("⚙️ Comparison Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            apply_noise_reduction = st.checkbox("Apply Noise Reduction", value=True)
            available_methods = list(models.keys()) + ['FastICA', 'PCA', 'NMF']
            selected_methods = st.multiselect(
                "Select methods to compare:",
                available_methods,
                default=available_methods[:2]
            )
        with col2:
            show_waveforms = st.checkbox("Show Waveform Plots", value=True)
            calculate_metrics = st.checkbox("Calculate Quality Metrics", value=True)
        
        if st.button("🚀 Start Comparison", type="primary"):
            if not selected_methods:
                st.error("Please select at least one method for comparison!")
                return
            
            separated_sources = {}
            processing_times = {}
            model_metrics = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, method_name in enumerate(selected_methods):
                status_text.text(f"Processing with {method_name}...")
                start_time = time.time()
                
                if method_name == 'FastICA':
                    audio = waveform.squeeze().numpy()
                    sources = apply_fastica(audio, sr)
                    src1, src2 = sources[0], sources[1]
                    model_size = "N/A (statistical)"
                elif method_name == 'PCA':
                    audio = waveform.squeeze().numpy()
                    sources = apply_pca(audio, sr)
                    src1, src2 = sources[0], sources[1]
                    model_size = "N/A (statistical)"
                elif method_name == 'NMF':
                    audio = waveform.squeeze().numpy()
                    sources = apply_nmf(audio, sr)
                    src1, src2 = sources[0], sources[1]
                    model_size = "N/A (statistical)"
                else:
                    model = models[method_name]
                    input_tensor = waveform.unsqueeze(0)
                    input_tensor = tensors_to_device(input_tensor, device="cpu")
                    model.to("cpu")
                    with torch.no_grad():
                        separated = model.separate(input_tensor)
                    src1 = separated[0, 0].cpu().numpy()
                    src2 = separated[0, 1].cpu().numpy()
                    # Amplify ConvTasNet outputs
                    gain = 1.5
                    src1 = src1 * gain
                    src2 = src2 * gain
                    # Normalize to prevent clipping
                    max_amplitude = max(np.max(np.abs(src1)), np.max(np.abs(src2)))
                    if max_amplitude > 1.0:
                        src1 = src1 / max_amplitude
                        src2 = src2 / max_amplitude
                    model_size = f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
                
                if apply_noise_reduction:
                    src1 = nr.reduce_noise(y=src1, sr=sr)
                    src2 = nr.reduce_noise(y=src2, sr=sr)
                
                processing_time = time.time() - start_time
                separated_sources[method_name] = (src1, src2)
                processing_times[method_name] = processing_time
                
                if calculate_metrics:
                    original_audio = waveform.squeeze().numpy()
                    si_sdr_src1 = calculate_si_sdr(original_audio[:len(src1)], src1)
                    si_sdr_src2 = calculate_si_sdr(original_audio[:len(src2)], src2)
                    pesq_src1 = calculate_pesq_approx(original_audio[:len(src1)], src1, sr)
                    pesq_src2 = calculate_pesq_approx(original_audio[:len(src2)], src2, sr)
                    model_metrics[method_name] = {
                        'si_sdr_src1': si_sdr_src1,
                        'si_sdr_src2': si_sdr_src2,
                        'avg_si_sdr': (si_sdr_src1 + si_sdr_src2) / 2,
                        'pesq_src1': pesq_src1,
                        'pesq_src2': pesq_src2,
                        'processing_time': processing_time,
                        'model_size': model_size
                    }
                
                progress_bar.progress((idx + 1) / len(selected_methods))
            
            # Store results in session state
            st.session_state.comparison_results = {
                'separated_sources': separated_sources,
                'processing_times': processing_times,
                'model_metrics': model_metrics,
                'selected_methods': selected_methods
            }
            status_text.text("✅ Comparison completed!")
        
        # Display results if available
        if st.session_state.comparison_results['separated_sources']:
            separated_sources = st.session_state.comparison_results['separated_sources']
            processing_times = st.session_state.comparison_results['processing_times']
            model_metrics = st.session_state.comparison_results['model_metrics']
            selected_methods = st.session_state.comparison_results['selected_methods']
            
            st.markdown("---")
            st.subheader("📊 Comparison Results")
            
            if calculate_metrics:
                st.markdown("#### 📈 Quality Metrics")
                metrics_df = create_metrics_dataframe(model_metrics)
                st.dataframe(metrics_df, use_container_width=True)
                if model_metrics:
                    best_model = max(model_metrics.keys(), 
                                   key=lambda x: model_metrics[x]['avg_si_sdr'])
                    st.success(f"🏆 Best performing model: **{best_model}** "
                              f"(Avg SI-SDR: {model_metrics[best_model]['avg_si_sdr']:.2f} dB)")
            
            if show_waveforms:
                st.markdown("#### 📊 Waveform Comparison")
                plot_comparison_waveforms(separated_sources, sr)
            
            st.markdown("#### 🎵 Audio Playback Comparison")
            for method_name, (src1, src2) in separated_sources.items():
                st.markdown(f"##### {method_name}")
                temp_src1 = tempfile.mktemp(suffix=f"_{method_name}_src1.wav")
                temp_src2 = tempfile.mktemp(suffix=f"_{method_name}_src2.wav")
                sf.write(temp_src1, src1, sr)
                sf.write(temp_src2, src2, sr)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.write("**Source 1:**")
                    st.audio(temp_src1, format="audio/wav")
                with col2:
                    st.write("**Source 2:**")
                    st.audio(temp_src2, format="audio/wav")
                with col3:
                    if calculate_metrics:
                        metrics = model_metrics[method_name]
                        st.metric("Avg SI-SDR", f"{metrics['avg_si_sdr']:.2f} dB")
                        st.metric("Processing Time", f"{metrics['processing_time']:.3f}s")
            
            st.markdown("---")
            st.markdown("### 💾 Download Separated Audio")
            download_method = st.selectbox(
                "Select method results to download:",
                selected_methods,
                key="download_select"
            )
            if download_method in separated_sources:
                src1, src2 = separated_sources[download_method]
                temp_download1 = tempfile.mktemp(suffix="_source1.wav")
                temp_download2 = tempfile.mktemp(suffix="_source2.wav")
                sf.write(temp_download1, src1, sr)
                sf.write(temp_download2, src2, sr)
                
                col1, col2 = st.columns(2)
                with col1:
                    with open(temp_download1, "rb") as file:
                        st.download_button(
                            label=f"📥 Download {download_method} - Source 1",
                            data=file.read(),
                            file_name=f"{download_method}_source1.wav",
                            mime="audio/wav"
                        )
                with col2:
                    with open(temp_download2, "rb") as file:
                        st.download_button(
                            label=f"📥 Download {download_method} - Source 2",
                            data=file.read(),
                            file_name=f"{download_method}_source2.wav",
                            mime="audio/wav"
                        )
    
    st.markdown("---")
    st.markdown("### ℹ️ About BSS Techniques")
    with st.expander("📚 Learn More About BSS Models"):
        st.markdown("""
        **ConvTasNet (Convolutional Time-domain Audio Separation Network)**:
        - Uses 1D convolutional neural networks
        - Operates directly in the time domain
        - Efficient and lightweight architecture
        
        **FastICA (Fast Independent Component Analysis)**:
        - Statistical method for blind source separation
        - Assumes sources are statistically independent
        - Computationally efficient but may struggle with complex mixtures
        
        **PCA (Principal Component Analysis)**:
        - Statistical method that transforms data to principal components
        - Reduces dimensionality to separate sources
        - Computationally efficient but may not handle non-linear mixtures well
        
        **NMF (Non-negative Matrix Factorization)**:
        - Decomposes non-negative data into non-negative components
        - Suitable for audio spectrogram separation
        - Effective for sources with distinct spectral patterns
        
        **Quality Metrics**:
        - **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio (higher is better)
        - **PESQ**: Perceptual Evaluation of Speech Quality (1-5 scale, higher is better)
        """)

if __name__ == "__main__":
    main()

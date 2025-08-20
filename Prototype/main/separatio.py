import streamlit as st
import torch
import torchaudio
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import sounddevice as sd
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

def plot_source_separation_steps(mixed_signal, separated_sources, sr, method="ConvTasNet"):
    """Plot step-by-step source separation process"""
    fig, axs = plt.subplots(3, 2, figsize=(16, 20))
    
    time = np.linspace(0, len(mixed_signal)/sr, len(mixed_signal))
    
    # Step 1: Original Mixed Signal
    axs[0,0].plot(time, mixed_signal, color='blue', linewidth=1)
    axs[0,0].set_title("Step 1: Original Mixed Signal - Time Domain", fontweight='bold', fontsize=14)
    axs[0,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Amplitude')
    axs[0,0].grid(True, alpha=0.3)
    axs[0,0].text(0.02, 0.95, 'INPUT', transform=axs[0,0].transAxes, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), 
                  fontsize=12, fontweight='bold')
    
    # Step 1: Mixed Signal Frequency Domain
    freqs = fftfreq(len(mixed_signal), 1/sr)
    X_mixed = fft(mixed_signal)
    axs[0,1].plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(X_mixed[:len(X_mixed)//2]) + 1e-10), 
                  color='blue', linewidth=1)
    axs[0,1].set_title("Step 1: Original Mixed Signal - Frequency Domain", fontweight='bold', fontsize=14)
    axs[0,1].set_xlabel('Frequency (Hz)')
    axs[0,1].set_ylabel('Magnitude (dB)')
    axs[0,1].grid(True, alpha=0.3)
    axs[0,1].text(0.02, 0.95, 'INPUT SPECTRUM', transform=axs[0,1].transAxes, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), 
                  fontsize=12, fontweight='bold')
    

    

    
    # Step 3: Separated Sources Time Domain
    time_sep = np.linspace(0, len(separated_sources[0])/sr, len(separated_sources[0]))
    
    axs[1,0].plot(time_sep, separated_sources[0], color='green', linewidth=1)
    axs[1,0].set_title("Step 3: Separated Source 1 - Time Domain", fontweight='bold', fontsize=14)
    axs[1,0].set_xlabel('Time (s)')
    axs[1,0].set_ylabel('Amplitude')
    axs[1,0].grid(True, alpha=0.3)
    axs[1,0].text(0.02, 0.95, 'OUTPUT 1', transform=axs[1,0].transAxes, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"), 
                  fontsize=12, fontweight='bold')
    
    axs[1,1].plot(time_sep, separated_sources[1], color='red', linewidth=1)
    axs[1,1].set_title("Step 3: Separated Source 2 - Time Domain", fontweight='bold', fontsize=14)
    axs[1,1].set_xlabel('Time (s)')
    axs[1,1].set_ylabel('Amplitude')
    axs[1,1].grid(True, alpha=0.3)
    axs[1,1].text(0.02, 0.95, 'OUTPUT 2', transform=axs[1,1].transAxes, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"), 
                  fontsize=12, fontweight='bold')
    
    # Step 4: Separated Sources Frequency Domain
    freqs_sep1 = fftfreq(len(separated_sources[0]), 1/sr)
    freqs_sep2 = fftfreq(len(separated_sources[1]), 1/sr)
    X_sep1 = fft(separated_sources[0])
    X_sep2 = fft(separated_sources[1])
    
    axs[2,0].plot(freqs_sep1[:len(freqs_sep1)//2], 20*np.log10(np.abs(X_sep1[:len(X_sep1)//2]) + 1e-10), 
                  color='green', linewidth=1)
    axs[2,0].set_title("Step 4: Separated Source 1 - Frequency Domain", fontweight='bold', fontsize=14)
    axs[2,0].set_xlabel('Frequency (Hz)')
    axs[2,0].set_ylabel('Magnitude (dB)')
    axs[2,0].grid(True, alpha=0.3)
    axs[2,0].text(0.02, 0.95, 'OUTPUT 1 SPECTRUM', transform=axs[2,0].transAxes, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"), 
                  fontsize=12, fontweight='bold')
    
    axs[2,1].plot(freqs_sep2[:len(freqs_sep2)//2], 20*np.log10(np.abs(X_sep2[:len(X_sep2)//2]) + 1e-10), 
                  color='red', linewidth=1)
    axs[2,1].set_title("Step 4: Separated Source 2 - Frequency Domain", fontweight='bold', fontsize=14)
    axs[2,1].set_xlabel('Frequency (Hz)')
    axs[2,1].set_ylabel('Magnitude (dB)')
    axs[2,1].grid(True, alpha=0.3)
    axs[2,1].text(0.02, 0.95, 'OUTPUT 2 SPECTRUM', transform=axs[2,1].transAxes, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"), 
                  fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_spectrogram_and_chroma(audio, sr, title="Spectrogram and Chroma Analysis"):
    """Plot spectrogram and chroma features"""
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    
    # Waveform
    time = np.linspace(0, len(audio)/sr, len(audio))
    axs[0].plot(time, audio, color='navy', linewidth=0.8)
    axs[0].set_title(f"{title} - Waveform", fontweight='bold', fontsize=14)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True, alpha=0.3)
    
    # Spectrogram
    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024, noverlap=512)
    im1 = axs[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    axs[1].set_title(f"{title} - Spectrogram", fontweight='bold', fontsize=14)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axs[1], label='Power (dB)')
    
    # Chroma
    try:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)
        chroma_time = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)
        chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        im2 = axs[2].pcolormesh(chroma_time, np.arange(12), chroma, shading='gouraud', cmap='plasma')
        axs[2].set_title(f"{title} - Chroma Features", fontweight='bold', fontsize=14)
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Chroma')
        axs[2].set_yticks(np.arange(12))
        axs[2].set_yticklabels(chroma_names)
        plt.colorbar(im2, ax=axs[2], label='Chroma Intensity')
    except Exception as e:
        axs[2].text(0.5, 0.5, f'Chroma analysis not available: {str(e)}', 
                   transform=axs[2].transAxes, ha='center', va='center')
        axs[2].set_title(f"{title} - Chroma Features (Not Available)", fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    return fig



def plot_separation_comparison(mixed_signal, separated_sources, sr):
    """Plot comparison between mixed and separated signals"""
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    
    time = np.linspace(0, len(mixed_signal)/sr, len(mixed_signal))
    time_sep = np.linspace(0, len(separated_sources[0])/sr, len(separated_sources[0]))
    
    # Mixed Signal
    axs[0].plot(time, mixed_signal, color='blue', linewidth=1, alpha=0.8)
    axs[0].set_title("Mixed Signal (Input)", fontweight='bold', fontsize=14)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True, alpha=0.3)
    axs[0].text(0.02, 0.95, 'MIXED', transform=axs[0].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), 
                fontsize=12, fontweight='bold')
    
    # Separated Source 1
    axs[1].plot(time_sep, separated_sources[0], color='green', linewidth=1, alpha=0.8)
    axs[1].set_title("Separated Source 1 (Output)", fontweight='bold', fontsize=14)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True, alpha=0.3)
    axs[1].text(0.02, 0.95, 'SOURCE 1', transform=axs[1].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"), 
                fontsize=12, fontweight='bold')
    
    # Separated Source 2
    axs[2].plot(time_sep, separated_sources[1], color='red', linewidth=1, alpha=0.8)
    axs[2].set_title("Separated Source 2 (Output)", fontweight='bold', fontsize=14)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True, alpha=0.3)
    axs[2].text(0.02, 0.95, 'SOURCE 2', transform=axs[2].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"), 
                fontsize=12, fontweight='bold')
    
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
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>🎵 Source Separation & Spectrogram Analysis</h1>
        <p>Visual Step-by-Step Source Separation with Spectrogram and Chroma Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        st.markdown("Upload an audio signal (WAV, MP3, or FLAC) containing mixed sources for separation.")
        
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
        st.markdown('<div class="section-header">🎙️ Record Audio Signal</div>', unsafe_allow_html=True)
        st.markdown("Record an audio signal containing mixed sources for separation.")
        
        duration = st.slider("Recording duration (seconds)", 1, 10, 3)
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### 🎧 Record Mixed Audio Signal")
        if st.button("🎙️ Record Signal", key="record"):
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
        st.markdown("**Mixed Signal**")
        temp_audio = tempfile.mktemp(suffix=".wav")
        sf.write(temp_audio, audio_data, sr)
        st.audio(temp_audio, format="audio/wav")
        
        st.markdown("---")
        
        # Perform Source Separation
        st.markdown('<div class="section-header">🛠️ Source Separation Process</div>', unsafe_allow_html=True)
        
        with st.spinner("🧠 Performing source separation..."):
            try:
                # Prepare single mono mixture
                waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                if sr != 16000:
                    st.info(f"Resampling from {sr} Hz to 16000 Hz for model compatibility.")
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                    sr = 16000
                
                input_tensor = tensors_to_device(waveform, device="cpu")
                model.to("cpu")
                with torch.no_grad():
                    separated = model.separate(input_tensor)
                
                src1 = separated[0, 0].cpu().numpy()
                src2 = separated[0, 1].cpu().numpy()
                
                # Apply noise reduction
                reduced_src1 = nr.reduce_noise(y=src1, sr=sr)
                reduced_src2 = nr.reduce_noise(y=src2, sr=sr)
                separated_sources = [reduced_src1, reduced_src2]

                reduced_src1 = reduced_src1 / (np.max(np.abs(reduced_src1)) + 1e-8)
                reduced_src2 = reduced_src2 / (np.max(np.abs(reduced_src2)) + 1e-8)

                separated_sources = [reduced_src1, reduced_src2]
                
                # Save separated sources
                sf.write("source1_clean.wav", reduced_src1, sr)
                sf.write("source2_clean.wav", reduced_src2, sr)
                
                st.success("✅ Source separation completed successfully!")
                
            except Exception as e:
                st.error(f"Error during source separation: {str(e)}")
                return
        
        # Create tabs for analysis
        tabs = st.tabs([
            "🔄 Separation Steps",
            "🎵 Audio Results", 
            "📊 Spectrogram & Chroma"
        ])
        
        # Tab 1: Step-by-step visualization
        with tabs[0]:
            st.markdown("### 🔍 Step-by-Step Source Separation Process")
            st.markdown("**Visual breakdown of how separation of mixed signals:**")
            
            fig_steps = plot_source_separation_steps(audio_data, separated_sources, sr, method="ConvTasNet")
            st.pyplot(fig_steps)
            
            # Comparison plot
            st.markdown("### 📈 Before vs After Comparison")
            fig_comparison = plot_separation_comparison(audio_data, separated_sources, sr)
            st.pyplot(fig_comparison)
        
        # Tab 2: Audio Results
        with tabs[1]:
            st.markdown("### 🎧 Separated Audio Sources")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🗣️ Separated Source 1**")
                st.audio("source1_clean.wav", format="audio/wav")
                st.download_button(
                    label="📥 Download Source 1",
                    data=open("source1_clean.wav", "rb").read(),
                    file_name="separated_source_1.wav",
                    mime="audio/wav"
                )
            with col2:
                st.markdown("**🗣️ Separated Source 2**")
                st.audio("source2_clean.wav", format="audio/wav")
                st.download_button(
                    label="📥 Download Source 2",
                    data=open("source2_clean.wav", "rb").read(),
                    file_name="separated_source_2.wav",
                    mime="audio/wav"
                )
            
            # Energy comparison
            st.markdown("### ⚡ Energy Analysis")
            energy1 = np.sum(separated_sources[0]**2)
            energy2 = np.sum(separated_sources[1]**2)
            energy_ratio = energy1 / (energy2 + 1e-10)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Source 1 Energy", f"{energy1:.2e}")
            with col2:
                st.metric("Source 2 Energy", f"{energy2:.2e}")
            with col3:
                st.metric("Energy Ratio", f"{energy_ratio:.2f}")
        
        # Tab 3: Spectrogram & Chroma Analysis
        with tabs[2]:
            st.markdown("### 🎵 Mixed Signal Analysis")
            fig_mixed = plot_spectrogram_and_chroma(audio_data, sr, "Mixed Signal")
            st.pyplot(fig_mixed)
            
           
            
            st.markdown("---")
            
            st.markdown("### 🎵 Separated Source 1 Analysis")
            fig_src1 = plot_spectrogram_and_chroma(separated_sources[0], sr, "Separated Source 1")
            st.pyplot(fig_src1)
            
          
            
            st.markdown("---")
            
            st.markdown("### 🎵 Separated Source 2 Analysis")
            fig_src2 = plot_spectrogram_and_chroma(separated_sources[1], sr, "Separated Source 2")
            st.pyplot(fig_src2)
            
         
        

if __name__ == "__main__":
    show_signals_systems_page()

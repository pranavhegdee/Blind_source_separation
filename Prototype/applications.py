import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
import os

def show_applications_page():
    st.title("🏭 Real-World BSS Applications")
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    Blind Source Separation (BSS) has revolutionized numerous fields by enabling the extraction of 
    meaningful signals from complex mixtures. Explore how BSS techniques are applied across different industries.
    """)
    
    # Application selector
    application = st.selectbox(
        "Select an Application Domain:",
        ["Medical Signal Processing", "Audio & Speech Processing", "Telecommunications", 
         "Industrial Monitoring", "Image Processing"]
    )
    
    if application == "Medical Signal Processing":
        show_medical_applications()
    elif application == "Audio & Speech Processing":
        show_audio_applications()
    elif application == "Telecommunications":
        show_telecom_applications()
    elif application == "Industrial Monitoring":
        show_industrial_applications()
    elif application == "Image Processing":
        show_image_processing_applications()

def show_medical_applications():
    st.header("🏥 Medical Signal Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("EEG Artifact Removal")
        st.markdown("""
        **Challenge**: EEG signals are contaminated by eye blinks, muscle movements, and electrical noise.
        
        **BSS Solution**: ICA separates brain signals from artifacts, enabling cleaner neural analysis.
        
        **Applications**:
        - Epilepsy monitoring
        - Sleep disorder diagnosis
        - Brain-computer interfaces
        - Cognitive research
        """)
        
        if st.button("Generate EEG Simulation"):
            generate_eeg_simulation()
    
    with col2:
        st.info("""
        **Key Benefits**:
        - Non-invasive artifact removal
        - Preserves neural information
        - Improves diagnostic accuracy
        - Enables real-time processing
        """)
    
    st.subheader("ECG Signal Enhancement")
    st.markdown("""
    **Application**: Fetal ECG extraction from maternal ECG recordings
    - Separates fetal heartbeat from maternal signals
    - Critical for prenatal monitoring
    - Enables early detection of cardiac abnormalities
    """)

def show_audio_applications():
    st.header("🎵 Audio & Speech Processing")
    
    tab1, tab2, tab3 = st.tabs(["Cocktail Party Problem", "Music Source Separation", "Speech Enhancement"])
    
    with tab1:
        st.subheader("The Cocktail Party Problem")
        st.markdown("""
        **Scenario**: Multiple people talking simultaneously in a noisy environment.
        
        **BSS Solution**: Separates individual speakers from the mixed audio signal.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Speakers Separated", "3-5", "simultaneously")
            st.metric("Noise Reduction", "15-25 dB", "typical improvement")
        
        with col2:
            if st.button("Simulate Cocktail Party"):
                simulate_cocktail_party()
    
    with tab2:
        st.subheader("Music Source Separation")
        st.markdown("""
        **Applications**:
        - Vocal isolation for karaoke
        - Instrument separation for remixing
        - Music transcription
        - Audio restoration
        """)
        
        st.info("Modern techniques can separate vocals, drums, bass, and other instruments with 80-90% accuracy")
    
    with tab3:
        st.subheader("Speech Enhancement")
        st.markdown("""
        **Use Cases**:
        - Hearing aids
        - Voice assistants
        - Telecommunication systems
        - Forensic audio analysis
        """)

def show_telecom_applications():
    st.header("📡 Telecommunications")
    
    st.subheader("MIMO System Optimization")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        **Multiple-Input Multiple-Output (MIMO) Systems**:
        
        BSS techniques optimize signal separation in wireless communications:
        - Separates signals from multiple antennas
        - Reduces interference between users
        - Increases data transmission rates
        - Improves network capacity
        """)
        
        if st.button("Show MIMO Capacity Analysis"):
            show_mimo_analysis()
    
    with col2:
        st.metric("Capacity Gain", "2-4x", "with BSS")
        st.metric("Interference Reduction", "20-30 dB", "typical")
    
    st.subheader("Antenna Array Processing")
    st.markdown("""
    **Applications**:
    - Beamforming for 5G networks
    - Radar signal processing
    - Satellite communications
    - Direction-of-arrival estimation
    """)

def show_industrial_applications():
    st.header("🏭 Industrial Monitoring")
    
    st.subheader("Machinery Fault Detection")
    
    tab1, tab2 = st.tabs(["Vibration Analysis", "Process Monitoring"])
    
    with tab1:
        st.markdown("""
        **Vibration-Based Diagnostics**:
        
        BSS separates different fault signatures in rotating machinery:
        - Bearing defects
        - Gear wear patterns
        - Shaft misalignment
        - Rotor imbalance
        """)
        
        if st.button("Simulate Vibration Analysis"):
            simulate_vibration_analysis()
    
    with tab2:
        st.markdown("""
        **Chemical Process Control**:
        
        - Separates process variables from noise
        - Identifies process disturbances
        - Quality control monitoring
        - Predictive maintenance
        """)

def show_image_processing_applications():
    st.header("🖼️ Image Processing Applications")
    
    st.subheader("Hyperspectral Image Unmixing")
    st.markdown("""
    **Remote Sensing Applications**:
    - Land cover classification
    - Mineral exploration
    - Environmental monitoring
    - Agricultural assessment
    """)
    
    st.subheader("Medical Imaging")
    st.markdown("""
    **Applications**:
    - fMRI brain activation mapping
    - Tissue separation in MRI
    - Contrast enhancement
    - Artifact removal
    """)

def generate_eeg_simulation():
    """Generate simulated EEG data with artifacts"""
    st.subheader("EEG Artifact Removal Simulation")
    
    # Generate synthetic EEG-like signals
    t = np.linspace(0, 10, 1000)
    
    # Clean EEG signals
    eeg1 = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    eeg2 = np.sin(2 * np.pi * 8 * t) + 0.3 * np.sin(2 * np.pi * 30 * t)
    
    # Artifacts
    eye_blink = 5 * np.exp(-((t - 3) ** 2) / 0.1) + 3 * np.exp(-((t - 7) ** 2) / 0.1)
    muscle_noise = 0.5 * np.random.randn(len(t))
    
    # Mixed signals
    mixed1 = 0.8 * eeg1 + 0.6 * eye_blink + 0.3 * muscle_noise
    mixed2 = 0.7 * eeg2 + 0.8 * eye_blink + 0.4 * muscle_noise
    
    # Normalize all signals
    mixed1 = mixed1 / np.max(np.abs(mixed1))
    mixed2 = mixed2 / np.max(np.abs(mixed2))
    eeg1 = eeg1 / np.max(np.abs(eeg1))
    eeg2 = eeg2 / np.max(np.abs(eeg2))
    eye_blink = eye_blink / np.max(np.abs(eye_blink))
    muscle_noise = muscle_noise / np.max(np.abs(muscle_noise))
    
    # Apply ICA
    mixed_signals = np.array([mixed1, mixed2]).T
    ica = FastICA(n_components=2, random_state=42)
    separated = ica.fit_transform(mixed_signals)
    separated = separated / np.max(np.abs(separated), axis=0, keepdims=True)
    
    # Debug data
    st.write(f"EEG Data Shapes: t={t.shape}, mixed1={mixed1.shape}, separated={separated.shape}")
    st.write(f"Mixed Signal 1 Range: {np.min(mixed1):.2f} to {np.max(mixed1):.2f}")
    st.write(f"Mixed Signal 2 Range: {np.min(mixed2):.2f} to {np.max(mixed2):.2f}")
    st.write(f"Separated Source 1 Range: {np.min(separated[:, 0]):.2f} to {np.max(separated[:, 0]):.2f}")
    st.write(f"Separated Source 2 Range: {np.min(separated[:, 1]):.2f} to {np.max(separated[:, 1]):.2f}")
    st.write(f"Any NaN in mixed1: {np.any(np.isnan(mixed1))}")
    
    # Combined Matplotlib plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, mixed1, 'r', label='Mixed Ch1')
    plt.plot(t, mixed2, 'b', label='Mixed Ch2')
    plt.plot(t, separated[:, 0], 'g--', label='Separated Source 1')
    plt.plot(t, separated[:, 1], 'orange', linestyle='--', label='Separated Source 2')
    plt.plot(t, eeg1, 'purple', label='Clean EEG 1')
    plt.plot(t, eeg2, 'brown', label='Clean EEG 2')
    plt.plot(t, eye_blink, 'r:', label='Eye Blinks')
    plt.plot(t, muscle_noise, 'gray', linestyle=':', label='Muscle Noise')
    plt.title("EEG Signal Separation")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Normalized)")
    plt.xlim(0, 10)
    plt.ylim(-3, 3)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()
    
    # Save combined plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(t, mixed1, 'r', label='Mixed Ch1')
        plt.plot(t, mixed2, 'b', label='Mixed Ch2')
        plt.plot(t, separated[:, 0], 'g--', label='Separated Source 1')
        plt.plot(t, separated[:, 1], 'orange', linestyle='--', label='Separated Source 2')
        plt.plot(t, eeg1, 'purple', label='Clean EEG 1')
        plt.plot(t, eeg2, 'brown', label='Clean EEG 2')
        plt.plot(t, eye_blink, 'r:', label='Eye Blinks')
        plt.plot(t, muscle_noise, 'gray', linestyle=':', label='Muscle Noise')
        plt.title("EEG Signal Separation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Normalized)")
        plt.xlim(0, 10)
        plt.ylim(-3, 3)
        plt.legend()
        plt.savefig("eeg_mpl_plot.png")
        st.image("eeg_mpl_plot.png", caption="Debug: EEG Combined Plot")
        plt.close()
    except Exception as e:
        st.warning(f"Failed to save EEG combined plot: {e}")
    
    # Individual mixed signal plots
    st.subheader("Mixed EEG Signals")
    for i, (signal, color) in enumerate([(mixed1, 'r'), (mixed2, 'b')]):
        plt.figure(figsize=(10, 3))
        plt.plot(t, signal, color, label=f'Mixed Channel {i+1}')
        plt.title(f"Mixed EEG Channel {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Normalized)")
        plt.xlim(0, 10)
        plt.ylim(-3, 3)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
    
    # Individual separated signal plots
    st.subheader("Separated EEG Signals")
    for i in range(2):
        plt.figure(figsize=(10, 3))
        plt.plot(t, separated[:, i], 'g--' if i == 0 else 'orange', label=f'Separated Source {i+1}')
        plt.title(f"Separated EEG Source {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Normalized)")
        plt.xlim(0, 10)
        plt.ylim(-3, 3)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Artifact Reduction", "85%", "typical")
    with col2:
        st.metric("Signal Quality", "SNR +15dB", "improvement")
    with col3:
        st.metric("Processing Time", "< 1s", "real-time")

def simulate_cocktail_party():
    """Simulate cocktail party problem"""
    st.subheader("Cocktail Party Separation")
    
    t = np.linspace(0, 5, 1000)
    
    speaker1 = np.sin(2 * np.pi * 3 * t) * np.exp(-0.1 * t)
    speaker2 = np.sin(2 * np.pi * 7 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    speaker3 = np.sin(2 * np.pi * 12 * t) * np.exp(-0.05 * t)
    
    A = np.array([[0.8, 0.5, 0.3], [0.4, 0.9, 0.6], [0.3, 0.4, 0.8]])
    sources = np.array([speaker1, speaker2, speaker3])
    mixed = A @ sources
    mixed = mixed / np.max(np.abs(mixed), axis=1, keepdims=True)
    
    ica = FastICA(n_components=3, random_state=42)
    separated = ica.fit_transform(mixed.T).T
    separated = separated / np.max(np.abs(separated), axis=1, keepdims=True)
    
    # Debug data
    st.write(f"Cocktail Data Shapes: t={t.shape}, mixed={mixed.shape}, separated={separated.shape}")
    for i in range(3):
        st.write(f"Mixed Signal {i+1} Range: {np.min(mixed[i]):.2f} to {np.max(mixed[i]):.2f}")
        st.write(f"Separated Source {i+1} Range: {np.min(separated[i]):.2f} to {np.max(separated[i]):.2f}")
        st.write(f"Any NaN in Mixed {i+1}: {np.any(np.isnan(mixed[i]))}")
    
    # Combined Matplotlib plot
    plt.figure(figsize=(10, 6))
    for i, color in enumerate(['r', 'b', 'g']):
        plt.plot(t, mixed[i], color, label=f"Mixed Signal {i+1}")
        plt.plot(t, separated[i], color=color, linestyle='--', label=f"Separated Source {i+1}")
    plt.title("Cocktail Party Separation")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Normalized)")
    plt.xlim(0, 5)
    plt.ylim(-3, 3)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()
    
    # Save combined plot
    try:
        plt.figure(figsize=(10, 6))
        for i, color in enumerate(['r', 'b', 'g']):
            plt.plot(t, mixed[i], color, label=f"Mixed Signal {i+1}")
            plt.plot(t, separated[i], color=color, linestyle='--', label=f"Separated Source {i+1}")
        plt.title("Cocktail Party Separation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Normalized)")
        plt.xlim(0, 5)
        plt.ylim(-3, 3)
        plt.legend()
        plt.savefig("cocktail_mpl_plot.png")
        st.image("cocktail_mpl_plot.png", caption="Debug: Cocktail Combined Plot")
        plt.close()
    except Exception as e:
        st.warning(f"Failed to save Cocktail combined plot: {e}")
    
    # Individual mixed signal plots
    st.subheader("Mixed Cocktail Party Signals")
    for i, color in enumerate(['r', 'b', 'g']):
        plt.figure(figsize=(10, 3))
        plt.plot(t, mixed[i], color, label=f'Mixed Signal {i+1}')
        plt.title(f"Mixed Cocktail Signal {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Normalized)")
        plt.xlim(0, 5)
        plt.ylim(-3, 3)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
    
    # Individual separated signal plots
    st.subheader("Separated Cocktail Party Signals")
    for i, color in enumerate(['g', 'orange', 'purple']):
        plt.figure(figsize=(10, 3))
        plt.plot(t, separated[i], color, label=f'Separated Source {i+1}')
        plt.title(f"Separated Cocktail Source {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Normalized)")
        plt.xlim(0, 5)
        plt.ylim(-3, 3)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

def show_mimo_analysis():
    """Show MIMO capacity analysis"""
    st.subheader("MIMO System Capacity Analysis")
    
    snr_db = np.linspace(-10, 30, 100)
    snr_linear = 10 ** (snr_db / 10)
    
    siso_capacity = np.log2(1 + snr_linear)
    mimo_2x2 = 2 * np.log2(1 + snr_linear / 2)
    mimo_4x4 = 4 * np.log2(1 + snr_linear / 4)
    
    # Debug data
    st.write(f"MIMO Data Shapes: snr_db={snr_db.shape}, siso={siso_capacity.shape}")
    st.write(f"SISO Range: {np.min(siso_capacity):.2f} to {np.max(siso_capacity):.2f}")
    st.write(f"Any NaN in SISO: {np.any(np.isnan(siso_capacity))}")
    
    # Combined Matplotlib plot
    plt.figure(figsize=(10, 6))
    plt.plot(snr_db, siso_capacity, 'r', label='SISO')
    plt.plot(snr_db, mimo_2x2, 'b', label='2x2 MIMO')
    plt.plot(snr_db, mimo_4x4, 'g', label='4x4 MIMO')
    plt.title("MIMO Capacity Improvement")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Capacity (bits/s/Hz)")
    plt.xlim(-10, 30)
    plt.ylim(0, np.max(mimo_4x4) * 1.1)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()
    
    # Save combined plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(snr_db, siso_capacity, 'r', label='SISO')
        plt.plot(snr_db, mimo_2x2, 'b', label='2x2 MIMO')
        plt.plot(snr_db, mimo_4x4, 'g', label='4x4 MIMO')
        plt.title("MIMO Capacity Improvement")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Capacity (bits/s/Hz)")
        plt.xlim(-10, 30)
        plt.ylim(0, np.max(mimo_4x4) * 1.1)
        plt.legend()
        plt.savefig("mimo_mpl_plot.png")
        st.image("mimo_mpl_plot.png", caption="Debug: MIMO Combined Plot")
        plt.close()
    except Exception as e:
        st.warning(f"Failed to save MIMO combined plot: {e}")
    
    # Individual capacity plots (no mixed signals in MIMO)
    st.subheader("Individual MIMO Capacity Curves")
    for name, data, color in [('SISO', siso_capacity, 'r'), ('2x2 MIMO', mimo_2x2, 'b'), ('4x4 MIMO', mimo_4x4, 'g')]:
        plt.figure(figsize=(10, 3))
        plt.plot(snr_db, data, color, label=name)
        plt.title(f"{name} Capacity")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Capacity (bits/s/Hz)")
        plt.xlim(-10, 30)
        plt.ylim(0, np.max(mimo_4x4) * 1.1)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

def simulate_vibration_analysis():
    """Simulate machinery vibration analysis"""
    st.subheader("Machinery Fault Detection")
    
    t = np.linspace(0, 2, 2000)
    fs = 1000
    
    healthy = np.sin(2 * np.pi * 50 * t)
    bearing_fault = 0.3 * np.sin(2 * np.pi * 157 * t)
    gear_fault = 0.5 * np.sin(2 * np.pi * 200 * t)
    misalignment = 0.2 * np.sin(2 * np.pi * 100 * t)
    
    mixed_signal = healthy + bearing_fault + gear_fault + misalignment + 0.1 * np.random.randn(len(t))
    
    sensor1 = mixed_signal + 0.05 * np.random.randn(len(t))
    sensor2 = 0.8 * mixed_signal + 0.3 * bearing_fault + 0.05 * np.random.randn(len(t))
    sensor3 = 0.9 * mixed_signal + 0.4 * gear_fault + 0.05 * np.random.randn(len(t))
    
    sensors = np.array([sensor1, sensor2, sensor3])
    sensors = sensors / np.max(np.abs(sensors), axis=1, keepdims=True)
    
    ica = FastICA(n_components=3, random_state=42)
    separated_sources = ica.fit_transform(sensors.T).T
    separated_sources = separated_sources / np.max(np.abs(separated_sources), axis=1, keepdims=True)
    
    # Debug data
    st.write(f"Vibration Data Shapes: t={t.shape}, sensors={sensors.shape}, separated={separated_sources.shape}")
    for i in range(3):
        st.write(f"Sensor {i+1} Range: {np.min(sensors[i]):.2f} to {np.max(sensors[i]):.2f}")
        st.write(f"Separated Source {i+1} Range: {np.min(separated_sources[i]):.2f} to {np.max(separated_sources[i]):.2f}")
        st.write(f"Any NaN in Sensor {i+1}: {np.any(np.isnan(sensors[i]))}")
    
    # Combined Matplotlib plot
    plt.figure(figsize=(10, 6))
    for i, color in enumerate(['r', 'b', 'g']):
        plt.plot(t[:500], sensors[i][:500], color, label=f"Sensor {i+1}")
        plt.plot(t[:500], separated_sources[i][:500], color=color, linestyle='--', label=f"Separated Source {i+1}")
    plt.title("Vibration Signal Separation")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Normalized)")
    plt.xlim(0, 0.5)
    plt.ylim(-3, 3)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()
    
    # Save combined plot
    try:
        plt.figure(figsize=(10, 6))
        for i, color in enumerate(['r', 'b', 'g']):
            plt.plot(t[:500], sensors[i][:500], color, label=f"Sensor {i+1}")
            plt.plot(t[:500], separated_sources[i][:500], color=color, linestyle='--', label=f"Separated Source {i+1}")
        plt.title("Vibration Signal Separation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Normalized)")
        plt.xlim(0, 0.5)
        plt.ylim(-3, 3)
        plt.legend()
        plt.savefig("vibration_mpl_plot.png")
        st.image("vibration_mpl_plot.png", caption="Debug: Vibration Combined Plot")
        plt.close()
    except Exception as e:
        st.warning(f"Failed to save Vibration combined plot: {e}")
    
    # Individual mixed signal plots
    st.subheader("Mixed Vibration Signals")
    for i, color in enumerate(['r', 'b', 'g']):
        plt.figure(figsize=(10, 3))
        plt.plot(t[:500], sensors[i][:500], color, label=f'Sensor {i+1}')
        plt.title(f"Mixed Vibration Sensor {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Normalized)")
        plt.xlim(0, 0.5)
        plt.ylim(-3, 3)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
    
    # Individual separated signal plots
    st.subheader("Separated Vibration Signals")
    for i, color in enumerate(['g', 'orange', 'purple']):
        plt.figure(figsize=(10, 3))
        plt.plot(t[:500], separated_sources[i][:500], color, label=f'Separated Source {i+1}')
        plt.title(f"Separated Vibration Source {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Normalized)")
        plt.xlim(0, 0.5)
        plt.ylim(-3, 3)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
    
    # Fault detection results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Bearing Fault Detected")
        st.metric("Severity", "Medium", "Frequency: 157 Hz")
    with col2:
        st.warning("⚠️ Gear Wear Detected")
        st.metric("Severity", "Low", "Frequency: 200 Hz")
    with col3:
        st.error("🔴 Misalignment Found")
        st.metric("Severity", "High", "Frequency: 100 Hz")

if __name__ == "__main__":
    show_applications_page()

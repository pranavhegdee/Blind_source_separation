import streamlit as st
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile

def show_icamix_page():
    st.title("🎛️ FastICA Mixing & Comparison")

    file1 = st.file_uploader("🎵 Upload Source 1", type=["wav"])
    file2 = st.file_uploader("🎵 Upload Source 2", type=["wav"])

    if file1 and file2:
        audio1, sr1 = torchaudio.load(file1)
        audio2, sr2 = torchaudio.load(file2)

        if sr1 != sr2:
            st.error("Sampling rates must match!")
            return

        audio1 = audio1.mean(0).numpy()
        audio2 = audio2.mean(0).numpy()
        T = min(len(audio1), len(audio2))
        audio1, audio2 = audio1[:T], audio2[:T]
        t = np.arange(T) / sr1

        rng = np.random.default_rng(42)
        A = rng.random((2, 2))
        S = np.vstack([audio1, audio2])
        X = A @ S

        st.subheader("🎧 Mixed Audio Playback")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as mixfile:
            sf.write(mixfile.name, X[0], sr1)
            st.audio(mixfile.name)

        X_centered = X - X.mean(axis=1, keepdims=True)
        cov = np.cov(X_centered)
        E, D, _ = np.linalg.svd(cov)
        whitening_matrix = np.linalg.inv(np.sqrt(np.diag(D))) @ E.T
        X_white = whitening_matrix @ X_centered

        N = 2
        W = np.zeros((N, N))
        for i in range(N):
            w = rng.random(N)
            w /= np.linalg.norm(w)
            for _ in range(500):
                wx = w @ X_white
                g = np.tanh(wx)
                g_prime = 1 - g**2
                w_new = (X_white @ g.T) / T - np.mean(g_prime) * w
                if i > 0:
                    w_new -= W[:i].T @ (W[:i] @ w_new)
                w_new /= np.linalg.norm(w_new)
                if np.linalg.norm(w_new - w) < 1e-6:
                    break
                w = w_new
            W[i] = w

        S_est = W @ X_white
        S1 = S_est[0] / np.max(np.abs(S_est[0]))
        S2 = S_est[1] / np.max(np.abs(S_est[1]))

        st.subheader("🔊 Separated Outputs")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as s1f:
            sf.write(s1f.name, S1, sr1)
            st.audio(s1f.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as s2f:
            sf.write(s2f.name, S2, sr1)
            st.audio(s2f.name)

        st.subheader("📊 All Waveforms")
        fig, axs = plt.subplots(4, 1, figsize=(12, 8))
        axs[0].plot(t, audio1); axs[0].set_title("Original Source 1")
        axs[1].plot(t, audio2); axs[1].set_title("Original Source 2")
        axs[2].plot(t, S1); axs[2].set_title("Separated Source 1")
        axs[3].plot(t, S2); axs[3].set_title("Separated Source 2")
        for ax in axs: ax.set_xlabel("Time (s)"); ax.grid(True)
        st.pyplot(fig)

        st.subheader("🌈 All Spectrograms")
        fig, axs = plt.subplots(2, 2, figsize=(12, 6))
        axs[0, 0].specgram(audio1, Fs=sr1, NFFT=1024, noverlap=512, cmap='plasma')
        axs[0, 0].set_title("Original Source 1")
        axs[0, 1].specgram(audio2, Fs=sr1, NFFT=1024, noverlap=512, cmap='plasma')
        axs[0, 1].set_title("Original Source 2")
        axs[1, 0].specgram(S1, Fs=sr1, NFFT=1024, noverlap=512, cmap='plasma')
        axs[1, 0].set_title("Separated Source 1")
        axs[1, 1].specgram(S2, Fs=sr1, NFFT=1024, noverlap=512, cmap='plasma')
        axs[1, 1].set_title("Separated Source 2")
        for ax in axs.flat:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
        plt.tight_layout()
        st.pyplot(fig)

        def calculate_metrics(ref, est):
            alpha = np.dot(est, ref) / np.dot(ref, ref)
            si_sdr = 10 * np.log10(np.sum((alpha * ref)**2) / np.sum((alpha * ref - est)**2))
            snr = 10 * np.log10(np.sum(ref**2) / np.sum((ref - est)**2))
            mse = np.mean((ref - est)**2)
            corr = np.corrcoef(ref, est)[0, 1]
            pesq = 1 + 4 * max(0, corr)
            return [si_sdr, snr, mse, corr, pesq]

        metrics_11 = calculate_metrics(audio1, S1)
        metrics_12 = calculate_metrics(audio1, S2)
        metrics_21 = calculate_metrics(audio2, S1)
        metrics_22 = calculate_metrics(audio2, S2)

        corr11 = metrics_11[3]; corr12 = metrics_12[3]
        corr21 = metrics_21[3]; corr22 = metrics_22[3]

        if corr11 + corr22 > corr12 + corr21:
            match_1 = ("Original 1", "Separated 1", metrics_11, audio1, S1)
            match_2 = ("Original 2", "Separated 2", metrics_22, audio2, S2)
        else:
            S1, S2 = S2, S1
            match_1 = ("Original 1", "Separated 2", metrics_12, audio1, S2)
            match_2 = ("Original 2", "Separated 1", metrics_21, audio2, S1)

        st.subheader("📈 Best Matched Comparisons")
        st.dataframe({
            "Metric": ["SI-SDR", "SNR", "MSE", "Correlation", "PESQ (approx)"],
            f"{match_1[0]} vs {match_1[1]}": match_1[2],
            f"{match_2[0]} vs {match_2[1]}": match_2[2]
        })

        st.subheader("📉 Metric Comparison")
        x = np.arange(len(match_1[2]))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - 0.2, match_1[2], 0.4, label=f'{match_1[0]} vs {match_1[1]}')
        ax.bar(x + 0.2, match_2[2], 0.4, label=f'{match_2[0]} vs {match_2[1]}')
        ax.set_xticks(x)
        ax.set_xticklabels(["SI-SDR", "SNR", "MSE", "Corr", "PESQ"])
        ax.legend()
        ax.grid(True)
        ax.set_title("Separated vs Original Metrics")
        st.pyplot(fig)







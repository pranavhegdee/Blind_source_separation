import streamlit as st
import numpy as np

def center(X):
    return X - np.mean(X, axis=0)

def whiten(X):
    from sklearn.decomposition import PCA
    pca = PCA(whiten=True)
    return pca.fit_transform(X), pca.components_

def g(u):
    return np.tanh(u)

def g_prime(u):
    return 1 - np.tanh(u) ** 2

def fastica(X, num_components, max_iter=200, tol=1e-5):
    st.info("Running FastICA algorithm ")
    X = center(X)
    X, _ = whiten(X)
    n, m = X.shape
    W = np.zeros((num_components, m))
    for i in range(num_components):
        w = np.random.rand(m)
        for iteration in range(max_iter):
            wx = np.dot(X, w)
            gwx = g(wx)
            g_wx = g_prime(wx)
            w_new = np.mean(X * gwx[:, np.newaxis], axis=0) - np.mean(g_wx) * w
            if i > 0:
                w_new -= W[:i].T @ (W[:i] @ w_new)
            w_new /= np.linalg.norm(w_new)
            if np.abs(np.abs(np.dot(w, w_new)) - 1) < tol:
                break
            w = w_new
        W[i, :] = w
    S = np.dot(X, W.T)
    return S

def show_home_page():
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Blind Source Separation with FastICA</h1>
        <p>Separate mixed audio signals using advanced algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="concept-card">
        <h2>🤔 What is FastICA?</h2>
        <p><strong>FastICA (Fast Independent Component Analysis)</strong> is a powerful algorithm used for blind source separation. 
        It's designed to separate mixed signals into their original independent components without knowing the mixing process.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 🔑 Key Concepts:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>🔍 Independent Component Analysis (ICA)</h4>
            <ul>
                <li>ICA assumes that the observed signals are linear mixtures of independent source signals</li>
                <li>The goal is to find a transformation that makes the output signals as statistically independent as possible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="concept-card">
            <h4>⚡ FastICA Algorithm</h4>
            <ul>
                <li>Uses a fixed-point iteration scheme for fast convergence</li>
                <li>Employs non-Gaussian maximization to find independent components</li>
                <li>Much faster than traditional gradient-based ICA methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("### ⚙️ How FastICA Works:")
    
    steps = [
        "**Centering**: Remove the mean from the data",
        "**Whitening**: Decorrelate the data and make it unit variance", 
        "**Iteration**: Use fixed-point algorithm to find independent components",
        "**Convergence**: Repeat until components are maximally independent"
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-card">
            {i}. {step}
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div class="concept-card">
        <h3>📐 Mathematical Foundation:</h3>
        <p>FastICA uses the following key equation for the fixed-point iteration:</p>
        <pre style="background: #f1f3f4; padding: 10px; border-radius: 5px; font-family: monospace;">
w+ = E{x g(w^T x)} - E{g'(w^T x)} w</pre>
        <p><strong>Where:</strong></p>
        <ul>
            <li><code>w</code> is the weight vector</li>
            <li><code>g()</code> is a non-linear function (typically tanh)</li>
            <li><code>g'()</code> is the derivative of g</li>
            <li><code>E{}</code> denotes expectation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🧠 FastICA: Python Implementation"):
        st.code("""
def fastica(X, num_components, max_iter=200, tol=1e-5):
    # Center the data
    X = center(X)
    
    # Whiten the data
    X, _ = whiten(X)
    
    n, m = X.shape
    W = np.zeros((num_components, m))
    
    for i in range(num_components):
        w = np.random.rand(m)
        
        for iteration in range(max_iter):
            # Fixed-point iteration
            wx = np.dot(X, w)
            gwx = g(wx)  # Non-linear function (tanh)
            g_wx = g_prime(wx)  # Derivative
            
            # Update rule
            w_new = np.mean(X * gwx[:, np.newaxis], axis=0) - np.mean(g_wx) * w
            
            # Orthogonalization (Gram-Schmidt)
            if i > 0:
                w_new -= W[:i].T @ (W[:i] @ w_new)
            
            # Normalize
            w_new /= np.linalg.norm(w_new)
            
            # Check convergence
            if np.abs(np.abs(np.dot(w, w_new)) - 1) < tol:
                break
                
            w = w_new
        
        W[i, :] = w
    
    # Compute separated sources
    S = np.dot(X, W.T)
    return S
        """, language="python")
    
    st.markdown("### 🚀 Applications:")
    
    applications = [
        "**Audio Processing**: Separating mixed audio signals (cocktail party problem)",
        "**Biomedical Signal Processing**: EEG, ECG signal separation",
        "**Image Processing**: Separating mixed images",
        "**Financial Data Analysis**: Finding independent market factors",
        "**Telecommunications**: Signal separation in communication systems"
    ]
    
    for app in applications:
        st.markdown(f"""
        <div class="application-item">
            {app}
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
        <h3>🆕 New Feature Alert!</h3>
        <p><strong>🎉 Dual Input Options Available!</strong></p>
        <p>🎤 <strong>Real-time Audio Separation:</strong> Record live audio and separate speakers instantly</p>
        <p>📁 <strong>File Upload Support:</strong> Upload pre-recorded audio files for separation</p>
        <p>🎓 <strong>Perfect for:</strong></p>
        <ul style="margin-left: 20px;">
            <li>📚 Separating voices in lecture recordings</li>
            <li>🎙️ Extracting individual speakers from meetings</li>
            <li>🎵 Isolating instruments from music tracks</li>
            <li>📞 Cleaning up conference call recordings</li>
            <li>🎬 Audio post-processing for videos</li>
        </ul>
        <p>💡 <strong>Pro Tip:</strong> For best results, use WAV files with clear audio quality!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer-style">
        <p>✨ <strong>Developed by:</strong> Pranav • Rakshaa • Sunidhi ✨</p>
        <p>🎵 Advanced Audio Signal Processing Team 🎵</p>
    </div>
    """, unsafe_allow_html=True)

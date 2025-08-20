import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.decomposition import FastICA
from skimage.transform import resize
from io import BytesIO
from PIL import Image
import seaborn as sns
from matplotlib.patches import Rectangle

def normalize_img(img):
    """Normalize image to [0, 1] range"""
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min)

def calculate_image_stats(img):
    """Calculate basic image statistics"""
    return {
        'mean': np.mean(img),
        'std': np.std(img),
        'min': np.min(img),
        'max': np.max(img),
        'shape': img.shape
    }

def create_comparison_plot(mixed_img1, mixed_img2, sep_img1, sep_img2):
    """Create an enhanced comparison plot with better styling"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FastICA Image Separation Results', fontsize=20, fontweight='bold', y=0.95)
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    images = [mixed_img1, mixed_img2, sep_img1, sep_img2]
    titles = ['Mixed Image 1', 'Mixed Image 2', 'Separated Image 1', 'Separated Image 2']
    
    for i, (ax, img, title, color) in enumerate(zip(axs.flat, images, titles, colors)):
        im = ax.imshow(img, cmap='gray', aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold', color=color, pad=20)
        ax.axis('off')
        
        rect = Rectangle((0, 0), img.shape[1]-1, img.shape[0]-1, 
                        linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    return fig

def create_histogram_comparison(mixed_img1, mixed_img2, sep_img1, sep_img2):
    """Create histogram comparison of pixel intensity distributions"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Pixel Intensity Distributions', fontsize=16, fontweight='bold')
    
    images = [mixed_img1, mixed_img2, sep_img1, sep_img2]
    titles = ['Mixed Image 1', 'Mixed Image 2', 'Separated Image 1', 'Separated Image 2']
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    
    for ax, img, title, color in zip(axs.flat, images, titles, colors):
        ax.hist(img.ravel(), bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def show_icaimage_page():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🖼 Advanced Image Separation using FastICA</h1>
        <p>Independent Component Analysis for Blind Source Separation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📊 About FastICA")
        st.info("""
        *Fast Independent Component Analysis* is a powerful technique for:
        - Blind source separation
        - Feature extraction
        - Noise reduction
        - Signal processing
        """)
        
        st.markdown("### ⚙ Parameters")
        n_components = st.slider("Number of Components", 2, 5, 2)
        random_state = st.number_input("Random State", 0, 100, 0)
        
        st.markdown("### 📈 Algorithm Info")
        st.markdown("""
        - *Algorithm*: FastICA
        - *Preprocessing*: Normalization
        - *Output*: Independent components
        """)
    
    # File upload section
    st.markdown("### 📁 Upload Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Mixed Image 1")
        file1 = st.file_uploader("Choose first mixed image", 
                                type=["png", "jpg", "jpeg"], 
                                key="img1",
                                help="Upload a grayscale or color image (will be converted to grayscale)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Mixed Image 2")
        file2 = st.file_uploader("Choose second mixed image", 
                                type=["png", "jpg", "jpeg"], 
                                key="img2",
                                help="Upload a grayscale or color image (will be converted to grayscale)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if file1 and file2:
        # Processing section
        with st.spinner('🔄 Processing images...'):
            # Convert uploaded files to grayscale numpy arrays
            img1 = Image.open(BytesIO(file1.read())).convert("L")
            img2 = Image.open(BytesIO(file2.read())).convert("L")
            
            mixed_img1 = np.array(img1, dtype=np.float32) / 255.0
            mixed_img2 = np.array(img2, dtype=np.float32) / 255.0
            
            # Resize if shapes don't match
            if mixed_img1.shape != mixed_img2.shape:
                mixed_img2 = resize(mixed_img2, mixed_img1.shape, anti_aliasing=True)
                st.warning(f"⚠ Images resized to match dimensions: {mixed_img1.shape}")
            
            # Prepare data for ICA
            X_mixed = np.stack([mixed_img1.ravel(), mixed_img2.ravel()], axis=0)
            
            # Apply FastICA
            ica = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
            S_estimated = ica.fit_transform(X_mixed.T).T
            
            # Reshape back to image format
            shape = mixed_img1.shape
            sep_img1 = normalize_img(S_estimated[0].reshape(shape))
            sep_img2 = normalize_img(S_estimated[1].reshape(shape))
        
        # Display image statistics
        st.markdown("### 📊 Image Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        stats1 = calculate_image_stats(mixed_img1)
        stats2 = calculate_image_stats(mixed_img2)
        stats3 = calculate_image_stats(sep_img1)
        stats4 = calculate_image_stats(sep_img2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4>Mixed Image 1</h4>
                <p>Shape: {stats1['shape']}</p>
                <p>Mean: {stats1['mean']:.3f}</p>
                <p>Std: {stats1['std']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h4>Mixed Image 2</h4>
                <p>Shape: {stats2['shape']}</p>
                <p>Mean: {stats2['mean']:.3f}</p>
                <p>Std: {stats2['std']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h4>Separated Image 1</h4>
                <p>Shape: {stats3['shape']}</p>
                <p>Mean: {stats3['mean']:.3f}</p>
                <p>Std: {stats3['std']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <h4>Separated Image 2</h4>
                <p>Shape: {stats4['shape']}</p>
                <p>Mean: {stats4['mean']:.3f}</p>
                <p>Std: {stats4['std']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Main results
        st.markdown("### 🎨 Separation Results")
        
        # Enhanced comparison plot
        fig_comparison = create_comparison_plot(mixed_img1, mixed_img2, sep_img1, sep_img2)
        st.pyplot(fig_comparison)
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Histogram Analysis", "🔍 Component Analysis", "💾 Download Results"])
        
        with tab1:
            st.markdown("#### Pixel Intensity Distribution Comparison")
            fig_hist = create_histogram_comparison(mixed_img1, mixed_img2, sep_img1, sep_img2)
            st.pyplot(fig_hist)
        
        with tab2:
            st.markdown("#### ICA Component Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="feature-card">
                    <h4>🔧 Algorithm Parameters</h4>
                    <p><strong>Components:</strong> {}</p>
                    <p><strong>Random State:</strong> {}</p>
                    <p><strong>Iterations:</strong> {}</p>
                </div>
                """.format(n_components, random_state, ica.n_iter_), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="feature-card">
                    <h4>📊 Mixing Matrix</h4>
                    <p>The estimated mixing matrix shows how the original sources were combined.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display mixing matrix
                mixing_matrix = ica.mixing_
                st.write("*Estimated Mixing Matrix:*")
                st.dataframe(mixing_matrix, use_container_width=True)
        
        with tab3:
            st.markdown("#### 💾 Download Separated Images")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Convert separated image 1 to PIL Image for download
                sep1_pil = Image.fromarray((sep_img1 * 255).astype(np.uint8))
                buf1 = BytesIO()
                sep1_pil.save(buf1, format="PNG")
                st.download_button(
                    label="📥 Download Separated Image 1",
                    data=buf1.getvalue(),
                    file_name="separated_image_1.png",
                    mime="image/png"
                )
            
            with col2:
                # Convert separated image 2 to PIL Image for download
                sep2_pil = Image.fromarray((sep_img2 * 255).astype(np.uint8))
                buf2 = BytesIO()
                sep2_pil.save(buf2, format="PNG")
                st.download_button(
                    label="📥 Download Separated Image 2",
                    data=buf2.getvalue(),
                    file_name="separated_image_2.png",
                    mime="image/png"
                )
        
        # Success message with animation
        st.balloons()
        st.success("✅ Image separation completed successfully using FastICA!")
        
        # Additional information
        with st.expander("ℹ How FastICA Works"):
            st.markdown("""
            *Independent Component Analysis (ICA)* is a computational method for separating a multivariate signal into additive subcomponents. FastICA specifically:
            
            1. *Preprocessing*: Centers and whitens the data
            2. *Optimization*: Uses a fixed-point algorithm for fast convergence
            3. *Independence*: Maximizes statistical independence between components
            4. *Separation*: Recovers original source signals from mixed observations
            
            This technique is particularly useful for:
            - 🎵 Audio source separation (cocktail party problem)
            - 🖼 Image processing and feature extraction
            - 📊 Dimensionality reduction
            - 🧠 EEG/MEG signal analysis
            """)



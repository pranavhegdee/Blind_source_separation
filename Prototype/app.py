import streamlit as st
from base.style import apply_styles
from main.home import show_home_page
from main.separatio import show_signals_systems_page
from main.comparison import show_comparison_page
from main.icamix import show_icamix_page
from main.icaimage import show_icaimage_page
from main.applications import show_applications_page

# Set page configuration
st.set_page_config(
    page_title="Real-Time BSS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles
apply_styles()

# Sidebar navigation
st.sidebar.markdown("## 🎵 Navigation")
page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "🎧 Source Separation", "⚖️ Model Comparison","🔊 Mixing and Separating using ICA","🖼 Image Separation","📈 Applications"])

# Page routing
if page == "🏠 Home":
    show_home_page()
elif page == "🎧 Source Separation":
    show_signals_systems_page()
elif page == "⚖️ Model Comparison":
    show_comparison_page()
elif page == "🔊 Mixing and Separating using ICA":
    show_icamix_page()
elif page == "🖼 Image Separation":
    show_icaimage_page()
elif page == "📈 Applications":
    show_applications_page() 


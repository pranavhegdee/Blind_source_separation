import streamlit as st

def apply_styles():
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .feature-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            color: white;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            border-left: 5px solid #4facfe;
        }
        
        .concept-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .step-card {
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            color: white;
            font-weight: bold;
        }
        
        .application-item {
            background: #e3f2fd;
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            border-left: 3px solid #2196f3;
        }
        
        .footer-style {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-top: 2rem;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
    </style>
    """, unsafe_allow_html=True)

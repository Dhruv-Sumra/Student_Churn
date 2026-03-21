"""
Lightweight launcher for the Streamlit app
This reduces initial memory load
"""
import sys
import os

# Set environment variables to reduce memory usage
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Run streamlit
if __name__ == '__main__':
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", "churn_app.py", "--server.maxUploadSize", "200"]
    sys.exit(stcli.main())

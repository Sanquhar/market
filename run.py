import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from market.gui.interface import main  

if __name__ == "__main__":
    main()

import streamlit as st

home_page = st.Page("Home.py", title = "Home")
method_page = st.Page("Underlying_Workflow.py", title = "Methodology")

pg = st.navigation([home_page, method_page])
pg.run()
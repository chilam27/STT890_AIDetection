import streamlit as st

home_page = st.Page("Home.py", title = "Home")
method_page = st.Page("Underlying_Workflow.py", title = "Methodology")
tab_monitor = st.Page("Monitoring.py", title = "Monitoring")

pg = st.navigation([home_page, method_page, tab_monitor])
pg.run()
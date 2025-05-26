import streamlit as st
from pages.login import login_page
from pages.app import app_page

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Define pages based on authentication status
if st.session_state.authenticated:
    # Show both pages when authenticated, but user starts on app page
    pages = [
        st.Page(app_page, title="App", icon="🎥"),
        st.Page(login_page, title="Logout", icon="🔒")  # Acts as logout when authenticated
    ]
else:
    # Only show login page when not authenticated
    pages = [
        st.Page(login_page, title="Login", icon="🔒")
    ]

# Create and run navigation
pg = st.navigation(pages)
pg.run()
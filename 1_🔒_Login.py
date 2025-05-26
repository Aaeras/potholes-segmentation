import streamlit as st
import bcrypt
from userdb import userList
st.set_page_config(page_title="Login", page_icon="🔒", initial_sidebar_state="collapsed")

def hashPassword(password):  # only including to show process, would not be in production
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify(name, password) -> bool: 
    if name in userList:
        if bcrypt.checkpw(password.encode('utf-8'), userList[name]):
            return True
        else:
            st.error("Incorrect Password")
            return False
    st.error('Incorrect Username')
    return False

def login_page():
    # Check if user is already authenticated (for logout functionality)
    if st.session_state.get("authenticated", False):
        st.title("Logout")
        st.write(f"Currently logged in as: {st.session_state.get('username', 'Unknown')}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.success("Successfully logged out!")
            st.rerun()  # Refresh to update navigation
        return
    
    # Login form
    st.markdown("<h1 style='text-align: center;'>Automated Pothole Detection Login</h1>", unsafe_allow_html=True)

    st.markdown("<h2>Group 20</h2>", unsafe_allow_html=True)
    st.markdown("<h3>Team Members:</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <table style='font-size:16px'>
        <tr><td>Craig Jones</td><td>24507358</td></tr>
        <tr><td>Warit Boonmasiri</td><td>25399522</td></tr>
        <tr><td>Yingrong Zhang</td><td>25428842</td></tr>
    </table>
    """, unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input('Password', type='password')

    if st.button("Login"):
        if verify(username, password):
            st.success(f"Welcome {username}! Please wait while you are redirected to our app!")
            st.session_state.authenticated = True
            st.session_state.username = username
            st.switch_page("pages/2_🎥_App.py")
if "authenticated" not in st.session_state:
    login_page()
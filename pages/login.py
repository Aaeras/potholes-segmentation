import streamlit as st;
import bcrypt;
from userdb import userList;

def hashPassword(password): #only including to show process, would not be in production
    return bcrypt.hashpw(password.encode('utf-8'),bcrypt.gensalt())

def verify(name, password) -> bool: 
    if (name in userList):
        if (bcrypt.checkpw(password.encode('utf-8'),userList[name])):
            return True
        else:
            st.error("Incorrect Password")
    return False

def login():
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
            st.success(f"Welcome {username}!")
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.switch_page(modelPage)
        else:
            st.error("Incorrect Username")
if "authenticated" in st.session_state:
    st.write(f"Currently logged in as {st.session_state['username']}")
    if (st.button("Logout")):
        st.session_state.clear()
    pg = st.navigation([loginPage, modelPage])
    pg.run()
else: 
    pg = st.navigation(loginPage)
    pg.run()

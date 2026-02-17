"""Authentication and role-based access control."""

import hashlib

import streamlit as st


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


USERS = {
    "admin": {
        "password_hash": _hash_password("changeme"),
        "role": "admin",
        "display_name": "Admin",
    },
    "viewer": {
        "password_hash": _hash_password("viewonly"),
        "role": "user",
        "display_name": "Viewer",
    },
}


def authenticate(username: str, password: str) -> dict | None:
    """Validate credentials. Returns user info dict or None."""
    user = USERS.get(username)
    if user and user["password_hash"] == _hash_password(password):
        return {"username": username, "role": user["role"], "display_name": user["display_name"]}
    return None


def render_login_page():
    """Render the login form. Sets session state on success."""
    st.header("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        user = authenticate(username, password)
        if user:
            st.session_state["authenticated"] = True
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("Invalid username or password.")


def is_authenticated() -> bool:
    return st.session_state.get("authenticated", False)


def get_user_role() -> str:
    return st.session_state.get("user", {}).get("role", "user")


def is_admin() -> bool:
    return get_user_role() == "admin"


def render_logout_button():
    """Display logged-in user info and a logout button in the sidebar."""
    user = st.session_state.get("user", {})
    st.sidebar.markdown(f"**{user.get('display_name', '')}** ({user.get('role', '')})")
    if st.sidebar.button("Log out"):
        st.session_state["authenticated"] = False
        st.session_state.pop("user", None)
        st.rerun()
    st.sidebar.divider()

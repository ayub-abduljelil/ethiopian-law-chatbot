"""
Authentication helpers — register, login, validation.
"""

import re
import bcrypt

from db_pgvector import create_user, get_user_by_username


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_USERNAME_RE = re.compile(r'^[A-Za-z0-9_]{3,30}$')
_EMAIL_RE    = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def _validate_username(username: str) -> str | None:
    if not username or not username.strip():
        return "Username cannot be empty."
    if not _USERNAME_RE.match(username.strip()):
        return "Username must be 3–30 characters: letters, digits, or underscores only."
    return None


def _validate_email(email: str) -> str | None:
    if not email or not email.strip():
        return "Email address cannot be empty."
    if not _EMAIL_RE.match(email.strip()):
        return "Please enter a valid email address."
    return None


def _validate_password(password: str) -> str | None:
    if not password:
        return "Password cannot be empty."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    if len(password) > 128:
        return "Password must be at most 128 characters."
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register(
    username: str,
    email:    str,
    password: str,
    confirm:  str,
) -> tuple[dict | None, str | None]:
    """
    Register a new user.

    Returns (user_dict, None) on success, or (None, error_message) on failure.
    user_dict keys: id, username, email.
    """
    username = username.strip()
    email    = email.strip().lower()

    err = _validate_username(username)
    if err: return None, err

    err = _validate_email(email)
    if err: return None, err

    err = _validate_password(password)
    if err: return None, err

    if password != confirm:
        return None, "Passwords do not match."

    if get_user_by_username(username):
        return None, f"Username '{username}' is already taken."

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        user_id = create_user(username, email, pw_hash)
    except Exception as exc:
        # Catch unique-constraint violation on email
        if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
            return None, "An account with this email address already exists."
        return None, f"Registration failed: {exc}"

    return {"id": user_id, "username": username, "email": email}, None


def login(username: str, password: str) -> tuple[dict | None, str | None]:
    """
    Authenticate an existing user.

    Returns (user_dict, None) on success, or (None, error_message) on failure.
    user_dict keys: id, username, email.
    """
    username = username.strip()

    if not username or not password:
        return None, "Please enter both username and password."

    row = get_user_by_username(username)
    if not row:
        return None, "Invalid username or password."

    if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return None, "Invalid username or password."

    return {"id": row["id"], "username": row["username"], "email": row.get("email", "")}, None

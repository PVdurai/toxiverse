"""
Security utilities for ToxiVerse.

Provides secure token generation and verification for user ID protection in URLs.
"""
from itsdangerous import URLSafeTimedSerializer
from flask import current_app


def get_serializer():
    """Get URL-safe serializer using app's SECRET_KEY."""
    return URLSafeTimedSerializer(current_app.config['SECRET_KEY'])


def generate_user_token(user_id):
    """
    Generate a secure token for a user ID.

    Args:
        user_id (int): User ID to encode

    Returns:
        str: Encrypted token string

    Example:
        >>> token = generate_user_token(26)
        >>> print(token)
        'eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoyNn0.xYz...'
    """
    serializer = get_serializer()
    return serializer.dumps({'user_id': user_id}, salt='user-access')


def verify_user_token(token, max_age=None):
    """
    Verify and decode a user token.

    Args:
        token (str): The encrypted token
        max_age (int, optional): Maximum age in seconds. None = no expiration

    Returns:
        int or None: User ID if valid, None if invalid/expired

    Example:
        >>> token = generate_user_token(26)
        >>> user_id = verify_user_token(token)
        >>> print(user_id)
        26
    """
    serializer = get_serializer()
    try:
        data = serializer.loads(token, salt='user-access', max_age=max_age)
        return data.get('user_id')
    except Exception:
        # Invalid token, expired, or tampered
        return None

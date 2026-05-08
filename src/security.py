import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass


PASSWORD_ALGORITHM = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 260000
LEGACY_SHA256_LENGTH = 64


@dataclass
class PasswordCheck:
    ok: bool
    message: str


def hash_password(password: str, *, iterations: int = PASSWORD_ITERATIONS) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    encoded = base64.b64encode(digest).decode("ascii")
    return f"{PASSWORD_ALGORITHM}${iterations}${salt}${encoded}"


def _verify_pbkdf2(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt, encoded = stored_hash.split("$", 3)
    except ValueError:
        return False
    if algorithm != PASSWORD_ALGORITHM:
        return False
    iterations = int(iterations_text)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    expected = base64.b64decode(encoded.encode("ascii"))
    return hmac.compare_digest(digest, expected)


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    if stored_hash.startswith(f"{PASSWORD_ALGORITHM}$"):
        return _verify_pbkdf2(password, stored_hash)

    # Backward-compatible verification for older SHA-256-only hashes.
    if len(stored_hash) == LEGACY_SHA256_LENGTH and all(character in "0123456789abcdef" for character in stored_hash.lower()):
        legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(legacy, stored_hash)
    return False


def password_needs_rehash(stored_hash: str) -> bool:
    if not stored_hash or not stored_hash.startswith(f"{PASSWORD_ALGORITHM}$"):
        return True
    try:
        _algorithm, iterations_text, _salt, _encoded = stored_hash.split("$", 3)
    except ValueError:
        return True
    try:
        return int(iterations_text) < PASSWORD_ITERATIONS
    except ValueError:
        return True


def validate_password_strength(password: str) -> PasswordCheck:
    if len(password) < 10:
        return PasswordCheck(False, "Password must be at least 10 characters.")
    if password.lower() == password or password.upper() == password:
        return PasswordCheck(False, "Password must mix upper and lower case characters.")
    if not any(character.isdigit() for character in password):
        return PasswordCheck(False, "Password must include at least one number.")
    if not any(not character.isalnum() for character in password):
        return PasswordCheck(False, "Password must include at least one special character.")
    return PasswordCheck(True, "Password strength is acceptable.")

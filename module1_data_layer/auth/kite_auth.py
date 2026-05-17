"""
SwingAdvisorBot — Module 1: Data Layer
auth/kite_auth.py — Kite Connect authentication manager

Kite Connect access tokens expire daily. Every trading day, the user
must complete a browser-based login flow to get a fresh request_token,
which is then exchanged for an access_token.

This module manages the entire lifecycle:
  1. Validate the current access_token by making a test API call.
  2. If expired → guide the user through re-authentication.
  3. Exchange request_token → access_token via kiteconnect SDK.
  4. Persist the new access_token to .env for the current session.
  5. Provide a valid KiteConnect client instance to all fetchers.

Design decisions:
  - Single KiteConnect instance shared across all fetchers (singleton).
  - Token validation via profile() call — lightweight and definitive.
  - .env update for access_token so subsequent runs reuse it same day.
  - All errors wrapped in KiteAuthError with clear guidance.
  - No credential hardcoding — everything from .env via Settings.

Flow for fetchers:
  from module1_data_layer.auth.kite_auth import kite_auth_manager
  kite = await kite_auth_manager.get_authenticated_client()
  # kite is a validated KiteConnect instance, ready for API calls.

Edge cases handled:
  - Token expired mid-session → detect via TokenException, re-validate.
  - Missing API key/secret → KiteAuthError with setup instructions.
  - Network failure during validation → KiteAuthError, not silent fail.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from kiteconnect import KiteConnect
from kiteconnect import exceptions as kite_exceptions

from module1_data_layer.cache import cache
from module1_data_layer.config import get_settings
from module1_data_layer.models import KiteAuthError

logger = logging.getLogger("swing_advisor.kite_auth")

# Path to .env file for access_token persistence
ENV_FILE_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


class KiteAuthManager:
    """Manages Kite Connect authentication lifecycle.

    Responsibilities:
      - Initialize KiteConnect with API key from Settings.
      - Validate access_token by calling Kite profile endpoint.
      - Exchange request_token for access_token when token expires.
      - Persist new access_token to .env file.
      - Provide a validated KiteConnect instance to all fetchers.

    Token lifecycle (daily):
      Morning:
        1. Bot starts, loads KITE_ACCESS_TOKEN from .env
        2. Calls validate_token() → makes profile() API call
        3. If valid → ready. If expired → triggers re-auth flow.

      Re-auth flow:
        1. Generate login_url for browser-based Zerodha login.
        2. User logs in, gets redirected with request_token in URL.
        3. User provides request_token to the bot.
        4. Bot exchanges request_token → access_token via Kite API.
        5. New access_token is saved to .env and cached.

    Usage:
        manager = KiteAuthManager()
        kite = await manager.get_authenticated_client()
        # kite.ltp("NSE:HDFCBANK")  → works
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._kite: KiteConnect | None = None
        self._is_validated: bool = False
        self._last_validation: datetime | None = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Create the KiteConnect instance with API key.

        Raises KiteAuthError if KITE_API_KEY is missing — the bot
        cannot function without Zerodha API credentials.
        """
        if not self._settings.kite_api_key:
            raise KiteAuthError(
                reason="KITE_API_KEY not found in .env file",
                suggestion=(
                    "Add KITE_API_KEY=<your_key> to .env. "
                    "Get your API key from https://developers.kite.trade/"
                ),
            )

        self._kite = KiteConnect(api_key=self._settings.kite_api_key)

        # Set access_token if available from .env
        if self._settings.kite_access_token:
            self._kite.set_access_token(self._settings.kite_access_token)
            logger.info(
                "Kite Connect initialized with existing access token from .env. "
                "Will validate on first API call."
            )
        else:
            logger.info(
                "Kite Connect initialized without access token. "
                "Re-authentication required before making API calls."
            )

    def validate_token(self) -> bool:
        """Validate the current access_token by calling Kite profile endpoint.

        Makes a lightweight API call to verify the token is still valid.
        Kite tokens expire daily, so this must be called at bot startup
        and periodically during long-running sessions.

        Returns:
            True if the token is valid and API calls will succeed.
            False if the token is expired/invalid and re-auth is needed.

        Does NOT raise exceptions — returns False on any failure.
        The caller decides whether to trigger re-auth or raise KiteAuthError.
        """
        if self._kite is None:
            return False

        try:
            profile = self._kite.profile()
            client_id = profile.get("user_id", "unknown")
            logger.info(
                f"Kite access token validated successfully. "
                f"Client: {client_id}. Session is active."
            )
            self._is_validated = True
            self._last_validation = datetime.now()
            return True

        except kite_exceptions.TokenException:
            logger.warning(
                "Kite access token has expired. "
                "Daily re-authentication required. "
                "Use get_login_url() to start the login flow."
            )
            self._is_validated = False
            return False

        except kite_exceptions.GeneralException as e:
            logger.warning(
                f"Kite token validation failed — API returned error: {e}. "
                f"Possible causes: network issue, Kite server downtime, or invalid API key."
            )
            self._is_validated = False
            return False

        except Exception as e:
            logger.warning(
                f"Unexpected error during Kite token validation: {type(e).__name__}: {e}. "
                f"Check network connectivity and Kite API status."
            )
            self._is_validated = False
            return False

    def get_login_url(self) -> str:
        """Generate the Kite Connect login URL for browser-based authentication.

        The user must:
          1. Open this URL in a browser.
          2. Log in with Zerodha credentials.
          3. Approve the API access.
          4. Copy the request_token from the redirect URL.
          5. Call set_access_token_from_request_token(request_token).

        Returns:
            The Kite Connect login URL string.

        Raises:
            KiteAuthError: If KiteConnect client is not initialized.
        """
        if self._kite is None:
            raise KiteAuthError(
                reason="KiteConnect client not initialized",
                suggestion="Ensure KITE_API_KEY is set in .env and call _initialize_client().",
            )

        login_url = self._kite.login_url()
        logger.info(
            f"Kite login URL generated. User must complete browser login "
            f"and provide the request_token from the redirect URL."
        )
        return login_url

    def set_access_token_from_request_token(self, request_token: str) -> str:
        """Exchange a request_token for an access_token via Kite API.

        This completes the daily re-authentication flow:
          1. Takes the request_token from the browser redirect.
          2. Calls Kite API to generate a session (access_token).
          3. Sets the access_token on the KiteConnect instance.
          4. Persists the new access_token to .env file.
          5. Invalidates all cached data (old token data is stale).

        Args:
            request_token: The token from the Kite login redirect URL.

        Returns:
            The new access_token string.

        Raises:
            KiteAuthError: If the exchange fails (invalid request_token,
                expired request_token, or network error).
        """
        if self._kite is None:
            raise KiteAuthError(
                reason="KiteConnect client not initialized",
                suggestion="Ensure KITE_API_KEY is set in .env.",
            )

        if not self._settings.kite_api_secret:
            raise KiteAuthError(
                reason="KITE_API_SECRET not found in .env file",
                suggestion=(
                    "Add KITE_API_SECRET=<your_secret> to .env. "
                    "Get your API secret from https://developers.kite.trade/"
                ),
            )

        if not request_token or not request_token.strip():
            raise KiteAuthError(
                reason="Empty request_token provided",
                suggestion=(
                    "Complete the browser login flow and copy the request_token "
                    "from the redirect URL query parameter."
                ),
            )

        try:
            session_data = self._kite.generate_session(
                request_token=request_token.strip(),
                api_secret=self._settings.kite_api_secret,
            )

            access_token = session_data["access_token"]
            self._kite.set_access_token(access_token)
            self._is_validated = True
            self._last_validation = datetime.now()

            # Persist to .env so subsequent runs within the same day reuse it
            self._persist_access_token(access_token)

            # Clear all cached data — old token data may be stale
            cache.clear()
            logger.info(
                f"Kite session established successfully. "
                f"Access token saved to .env. All caches cleared. "
                f"Client: {session_data.get('user_id', 'unknown')}."
            )

            return access_token

        except kite_exceptions.TokenException as e:
            raise KiteAuthError(
                reason=f"Request token exchange failed: {e}",
                suggestion=(
                    "The request_token may have expired (valid for ~2 minutes). "
                    "Complete the login flow again and provide a fresh request_token."
                ),
            ) from e

        except kite_exceptions.GeneralException as e:
            raise KiteAuthError(
                reason=f"Kite API error during session generation: {e}",
                suggestion="Check KITE_API_KEY and KITE_API_SECRET in .env. Retry login flow.",
            ) from e

        except KeyError:
            raise KiteAuthError(
                reason="Session response missing access_token field",
                suggestion="Kite API returned unexpected response format. Contact Zerodha support.",
            )

    def _persist_access_token(self, access_token: str) -> None:
        """Save the new access_token to the .env file.

        Reads the existing .env, replaces or appends KITE_ACCESS_TOKEN,
        and writes it back. This ensures the token survives bot restarts
        within the same trading day.

        Args:
            access_token: The new access token to persist.
        """
        try:
            env_content = ""
            if ENV_FILE_PATH.exists():
                env_content = ENV_FILE_PATH.read_text(encoding="utf-8")

            # Replace existing KITE_ACCESS_TOKEN line or append
            new_line = f"KITE_ACCESS_TOKEN={access_token}"
            lines = env_content.splitlines()
            found = False

            for i, line in enumerate(lines):
                if line.startswith("KITE_ACCESS_TOKEN="):
                    lines[i] = new_line
                    found = True
                    break

            if not found:
                lines.append(new_line)

            ENV_FILE_PATH.write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

            # Also update the environment variable for current process
            os.environ["KITE_ACCESS_TOKEN"] = access_token

            logger.info(
                f"Access token persisted to {ENV_FILE_PATH}. "
                f"Current session and future restarts will use this token."
            )

        except OSError as e:
            logger.warning(
                f"Could not persist access token to .env: {e}. "
                f"Token is active in memory but will be lost on restart. "
                f"Manually add KITE_ACCESS_TOKEN={access_token} to .env."
            )

    async def get_authenticated_client(self) -> KiteConnect:
        """Get a validated KiteConnect instance ready for API calls.

        This is the primary method all fetchers call. It ensures the
        returned KiteConnect instance has a valid, tested access_token.

        Flow:
          1. If already validated this session → return immediately.
          2. If not validated → call validate_token().
          3. If validation passes → return client.
          4. If validation fails → raise KiteAuthError with re-auth instructions.

        Returns:
            A KiteConnect instance with a validated access_token.

        Raises:
            KiteAuthError: If the token is invalid and re-authentication
                is required. The error message includes the login URL.
        """
        if self._kite is None:
            raise KiteAuthError(
                reason="KiteConnect client not initialized",
                suggestion="Check KITE_API_KEY in .env file.",
            )

        if self._is_validated:
            return self._kite

        if self.validate_token():
            return self._kite

        login_url = self.get_login_url()
        raise KiteAuthError(
            reason="Kite access token expired or invalid — daily re-authentication required",
            suggestion=(
                f"Complete the login flow:\n"
                f"  1. Open this URL in browser: {login_url}\n"
                f"  2. Log in with Zerodha credentials.\n"
                f"  3. Copy the request_token from the redirect URL.\n"
                f"  4. Call: kite_auth_manager.set_access_token_from_request_token('<token>')"
            ),
        )

    def invalidate(self) -> None:
        """Mark the current token as invalid.

        Called by fetchers when they catch a TokenException mid-session.
        Forces the next get_authenticated_client() call to re-validate.
        """
        self._is_validated = False
        logger.info(
            "Kite access token marked as invalid. "
            "Next API call will trigger re-validation."
        )

    @property
    def is_authenticated(self) -> bool:
        """Whether the current session has a validated access_token."""
        return self._is_validated

    @property
    def client_id(self) -> str:
        """Zerodha client ID from settings."""
        return self._settings.kite_client_id

    @property
    def status_report(self) -> dict[str, str | bool | None]:
        """Authentication status for pipeline health reporting."""
        return {
            "is_authenticated": self._is_validated,
            "last_validation": (
                self._last_validation.isoformat() if self._last_validation else None
            ),
            "client_id": self._settings.kite_client_id,
            "has_api_key": bool(self._settings.kite_api_key),
            "has_api_secret": bool(self._settings.kite_api_secret),
            "has_access_token": bool(self._settings.kite_access_token),
        }


# ─────────────────────────────────────────────────────────────
# Module-level singleton — all fetchers share one auth manager
# Import as: from module1_data_layer.auth.kite_auth import kite_auth_manager
# ─────────────────────────────────────────────────────────────

kite_auth_manager = KiteAuthManager()

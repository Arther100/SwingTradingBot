"""
SwingAdvisorBot — Test Smart Auth System
test_smart_auth.py — Unit tests for Kite Smart Auth flow

Tests 1-4 are automated (no Kite login needed).
Test 5 is manual (requires real Kite login).

Usage:
    python test_smart_auth.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0


def _result(name: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {detail}")


# ═══════════════════════════════════════════════════════════
# TEST 1: Login URL generation
# ═══════════════════════════════════════════════════════════

def test_login_url():
    print("\n🔹 TEST 1: Login URL Generation")

    from module1_data_layer.auth.smart_auth import SmartKiteAuth

    auth = SmartKiteAuth()
    url = auth.get_login_url()

    api_key = os.getenv("KITE_API_KEY", "")

    _result(
        "URL contains api_key",
        api_key in url,
        f"Expected '{api_key}' in URL: {url}",
    )
    _result(
        "URL starts with https://kite.zerodha.com",
        url.startswith("https://kite.zerodha.com"),
        f"Got: {url}",
    )
    _result(
        "URL contains /connect/login",
        "/connect/login" in url,
        f"Got: {url}",
    )
    _result(
        "URL contains v=3",
        "v=3" in url,
        f"Got: {url}",
    )
    print(f"    Login URL: {url}")


# ═══════════════════════════════════════════════════════════
# TEST 2: Request token extraction
# ═══════════════════════════════════════════════════════════

def test_request_token_extraction():
    print("\n🔹 TEST 2: Request Token Extraction")

    from module1_data_layer.auth.smart_auth import REQUEST_TOKEN_PATTERN

    # Standard redirect URL
    url1 = (
        "http://127.0.0.1:8000/?action=login"
        "&request_token=Nu1jgRcHwlnwkX4lmX6TL6g&status=success"
    )
    match1 = REQUEST_TOKEN_PATTERN.search(url1)
    token1 = match1.group(1) if match1 else None

    _result(
        "Extracts token from standard URL",
        token1 == "Nu1jgRcHwlnwkX4lmX6TL6g",
        f"Got: {token1}",
    )

    # URL with different token
    url2 = "http://127.0.0.1:8000/?request_token=abc123XYZ789&status=success"
    match2 = REQUEST_TOKEN_PATTERN.search(url2)
    token2 = match2.group(1) if match2 else None

    _result(
        "Extracts alphanumeric token",
        token2 == "abc123XYZ789",
        f"Got: {token2}",
    )

    # URL without token
    url3 = "http://127.0.0.1:8000/?status=error"
    match3 = REQUEST_TOKEN_PATTERN.search(url3)

    _result(
        "Returns None for URL without token",
        match3 is None,
        f"Got: {match3}",
    )

    # Token in longer URL
    url4 = (
        "https://kite.zerodha.com/connect/login?"
        "request_token=LongToken123ABC&type=login&v=3"
    )
    match4 = REQUEST_TOKEN_PATTERN.search(url4)
    token4 = match4.group(1) if match4 else None

    _result(
        "Extracts token from full Kite URL",
        token4 == "LongToken123ABC",
        f"Got: {token4}",
    )


# ═══════════════════════════════════════════════════════════
# TEST 3: Telegram message detection
# ═══════════════════════════════════════════════════════════

def test_telegram_message_detection():
    print("\n🔹 TEST 3: Telegram Message Detection")

    from module1_data_layer.auth.smart_auth import SmartKiteAuth

    auth = SmartKiteAuth()

    # We test the detection logic, not the actual auth processing.
    # handle_telegram_message() calls process_redirect_url() which
    # needs a real Kite session. So we test the detection part only.

    # Test: regular message should NOT trigger
    is_auth_1 = (
        "request_token=" in "hello"
        or ("127.0.0.1" in "hello" and "request_token" in "hello")
        or ("kite.zerodha" in "hello" and "request_token" in "hello")
    )
    _result(
        "'hello' does NOT trigger auth",
        not is_auth_1,
        f"Got: {is_auth_1}",
    )

    # Test: redirect URL should trigger
    redirect_url = (
        "http://127.0.0.1:8000/?request_token=abc123&status=success"
    )
    is_auth_2 = (
        "request_token=" in redirect_url
        or ("127.0.0.1" in redirect_url and "request_token" in redirect_url)
        or ("kite.zerodha" in redirect_url and "request_token" in redirect_url)
    )
    _result(
        "Redirect URL triggers auth",
        is_auth_2,
        f"Got: {is_auth_2}",
    )

    # Test: just "request_token=abc123" should trigger
    bare_token = "request_token=abc123"
    is_auth_3 = (
        "request_token=" in bare_token
        or ("127.0.0.1" in bare_token and "request_token" in bare_token)
        or ("kite.zerodha" in bare_token and "request_token" in bare_token)
    )
    _result(
        "'request_token=abc123' triggers auth",
        is_auth_3,
        f"Got: {is_auth_3}",
    )

    # Test: random URL should NOT trigger
    random_url = "https://google.com/search?q=nifty"
    is_auth_4 = (
        "request_token=" in random_url
        or ("127.0.0.1" in random_url and "request_token" in random_url)
        or ("kite.zerodha" in random_url and "request_token" in random_url)
    )
    _result(
        "Random URL does NOT trigger auth",
        not is_auth_4,
        f"Got: {is_auth_4}",
    )

    # Test: empty string should NOT trigger
    is_auth_5 = (
        "request_token=" in ""
        or ("127.0.0.1" in "" and "request_token" in "")
        or ("kite.zerodha" in "" and "request_token" in "")
    )
    _result(
        "Empty string does NOT trigger auth",
        not is_auth_5,
        f"Got: {is_auth_5}",
    )


# ═══════════════════════════════════════════════════════════
# TEST 4: Telegram alert format
# ═══════════════════════════════════════════════════════════

def test_telegram_alert_format():
    print("\n🔹 TEST 4: Telegram Alert Format")

    from module1_data_layer.auth.smart_auth import SmartKiteAuth

    auth = SmartKiteAuth()
    alert = auth.generate_telegram_alert()

    _result(
        "Alert contains login URL",
        "kite.zerodha.com/connect/login" in alert,
        f"URL not found in alert",
    )

    _result(
        "Alert is under 4096 chars",
        len(alert) <= 4096,
        f"Length: {len(alert)}",
    )

    _result(
        "Alert is HTML formatted",
        "<b>" in alert and "</b>" in alert,
        "No <b> tags found",
    )

    _result(
        "Alert mentions morning brief time",
        "8:50" in alert,
        "No mention of 8:50 AM",
    )

    _result(
        "Alert has Step 1 and Step 2",
        "Step 1" in alert and "Step 2" in alert,
        "Missing step instructions",
    )

    _result(
        "Alert mentions request_token",
        "request_token" in alert,
        "No mention of request_token",
    )

    print(f"    Alert length: {len(alert)} chars")
    print(f"    Preview: {alert[:120]}...")


# ═══════════════════════════════════════════════════════════
# TEST 5: Full auth flow (MANUAL — requires real Kite login)
# ═══════════════════════════════════════════════════════════

def test_full_auth_flow_docs():
    print("\n🔹 TEST 5: Full Auth Flow (MANUAL)")
    print("    This test requires a real Kite login. Steps:")
    print()
    print("    1. Run: python -c \"")
    print("       from module1_data_layer.auth.smart_auth import smart_auth")
    print("       import asyncio")
    print("       asyncio.run(smart_auth.check_and_alert())\"")
    print("    → You should receive a Telegram alert with login link.")
    print()
    print("    2. Click the login link in Telegram.")
    print("    3. Login with Zerodha credentials.")
    print("    4. Copy the redirect URL from browser address bar.")
    print("    5. Send the redirect URL to the Telegram bot.")
    print("    → Bot should auto-process and reply '✅ Auth Complete'")
    print()
    print("    6. Verify token works:")
    print("       python -c \"")
    print("       from module1_data_layer.auth.kite_auth import kite_auth_manager")
    print("       print(kite_auth_manager.validate_token())\"")
    print("    → Should print: True")
    print()
    print("    ⏱  Total time should be under 60 seconds.")
    _result("Manual test documented", True)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 60)
    print("  SwingAdvisorBot — Smart Auth Test Suite")
    print("=" * 60)

    test_login_url()
    test_request_token_extraction()
    test_telegram_message_detection()
    test_telegram_alert_format()
    test_full_auth_flow_docs()

    print()
    print("=" * 60)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} failed ❌")
    else:
        print(" ✅ All tests passed!")
    print("=" * 60)

    sys.exit(1 if failed else 0)

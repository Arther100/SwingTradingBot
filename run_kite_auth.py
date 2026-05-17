import os
from dotenv import load_dotenv, set_key
load_dotenv()

from kiteconnect import KiteConnect


def generate_access_token(request_token: str) -> str:
    """
    Exchanges request_token for access_token.
    request_token: temporary (2 min expiry)
    access_token: daily token (6 AM IST expiry)
    """
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError(
            "KITE_API_KEY and KITE_API_SECRET "
            "must be set in .env"
        )

    kite = KiteConnect(api_key=api_key)

    # Exchange request_token for access_token
    session = kite.generate_session(
        request_token=request_token,
        api_secret=api_secret
    )

    access_token = session["access_token"]

    # Save access_token to .env
    env_path = os.path.join(
        os.path.dirname(__file__), ".env"
    )
    set_key(env_path, "KITE_ACCESS_TOKEN", access_token)

    print(f"✅ Access token generated successfully")
    print(f"   Token: {access_token[:10]}... (saved to .env)")
    return access_token


if __name__ == "__main__":
    print("=== KITE TOKEN REFRESH ===")
    print("Paste your request_token from redirect URL:")
    request_token = input("request_token: ").strip()

    if not request_token:
        print("❌ No token provided")
        exit(1)

    try:
        token = generate_access_token(request_token)
        print("✅ .env updated with new access token")
        print("✅ Ready to run smoke_test.py")
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("Get a new request_token — they expire in 2 minutes")

import httpx
import os
from dotenv import load_dotenv
load_dotenv()

async def test_telegram():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print("=== TELEGRAM CONNECTION TEST ===")
    print(f"Token: {token[:10]}... (first 10 chars)")
    print(f"Chat ID: {chat_id}")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "✅ SwingAdvisorBot connected!\n\nYour AI senior finance advisor is ready.",
                "parse_mode": "HTML"
            },
            timeout=10.0
        )
        data = response.json()

        if data.get("ok"):
            msg_id = data["result"]["message_id"]
            print(f"✅ Message sent successfully")
            print(f"   Message ID: {msg_id}")
            print(f"   Check your Telegram now!")
        else:
            print(f"❌ Failed: {data.get('description')}")

import asyncio
asyncio.run(test_telegram())

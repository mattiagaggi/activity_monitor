#!/usr/bin/env python3
"""
Simple script to test the CLAUDE_API_KEY from .env file
"""
import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

def test_claude_api():
    api_key = os.getenv('CLAUDE_API_KEY')

    if not api_key:
        print("❌ Error: CLAUDE_API_KEY not found in environment variables")
        return False

    print(f"✓ API Key found: {api_key[:20]}...")
    print("\nTesting API connection...")

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Make a simple test request
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Say 'Hello! The API key is working correctly.' and nothing else."}
            ]
        )

        response_text = message.content[0].text
        print(f"\n✓ API Response received:")
        print(f"  {response_text}")
        print(f"\n✓ Token usage: {message.usage.input_tokens} input, {message.usage.output_tokens} output")
        print("\n✅ API key is valid and working!")
        return True

    except anthropic.AuthenticationError:
        print("\n❌ Authentication Error: The API key is invalid or expired")
        return False
    except anthropic.PermissionDeniedError:
        print("\n❌ Permission Denied: The API key doesn't have access to this resource")
        return False
    except anthropic.RateLimitError:
        print("\n❌ Rate Limit: Too many requests")
        return False
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    test_claude_api()

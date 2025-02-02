import re
import json
import os
import asyncio

from dotenv import load_dotenv
load_dotenv()

import instaloader

# Use the new OpenAI client
from openai import OpenAI
client = OpenAI()


def parse_instagram_username(profile_url: str) -> str:
    """
    Extracts the Instagram username from a given profile URL.
    Example: "https://www.instagram.com/myusername/" --> "myusername"
    """
    match = re.search(r"instagram\.com/([^/?#]+)", profile_url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Could not extract username from the URL.")


def analyze_instagram_profile(profile_url: str, context_filename="instagram_context.txt") -> str:
    """
    Scrapes the Instagram profile’s bio and posts using Instaloader.
    If the context file already exists, it is reused instead of fetching again.
    Writes (or reuses) a text file containing the bio and all posts.
    """
    if os.path.exists(context_filename):
        print("[INFO] Context file already exists; using the existing file.")
        return context_filename

    username = parse_instagram_username(profile_url)
    print(f"[INFO] Extracted username: {username}")

    loader = instaloader.Instaloader(
        download_pictures=False,
        download_video_thumbnails=False,
        save_metadata=False,
        download_comments=False
    )
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
    except Exception as e:
        raise RuntimeError(f"Error loading profile: {e}")

    bio = profile.biography
    posts_data = []
    print("[INFO] Retrieving posts... (this might take a minute)")
    for post in profile.get_posts():
        posts_data.append({
            'caption': post.caption or "",
            'url': f"https://www.instagram.com/p/{post.shortcode}/"
        })

    with open(context_filename, "w", encoding="utf-8") as f:
        f.write("=== Instagram Profile Context ===\n")
        f.write(f"Username: {username}\n")
        f.write(f"Bio: {bio}\n\n")
        f.write("=== Past Posts ===\n")
        for post in posts_data:
            f.write(f"Post URL: {post['url']}\n")
            f.write(f"Caption: {post['caption']}\n")
            f.write("-" * 40 + "\n")
    
    print(f"[INFO] Instagram context saved in {context_filename}")
    return context_filename


def generate_new_posts(context_filename: str, num_posts: int = 5) -> list:
    """
    Uses OpenAI's Chat API to generate new Instagram posts based on the context.
    Each post is structured as follows:
      - Hook: A one-sentence attention-grabbing hook.
      - Content Pages: Three pages (each with a title and a body).
      - CTA: A call-to-action sentence.
    
    Returns a list of post objects in JSON format.
    """
    with open(context_filename, "r", encoding="utf-8") as f:
        context_text = f.read()

    prompt = f"""
You are an expert copywriter and content creator for Instagram. You have analyzed my past posts (see below)
and understand the editorial line, tone, and style. Now, generate {num_posts} new Instagram posts that fit 
the same style and that have not been published before.

Each post should be structured as follows:

- **Hook**: A one-sentence attention-grabbing hook.
- **Content Pages**: Three pages. Each page must have:
    - A **Title**
    - A **Body** (an explanation or insight)
- **CTA**: A final page with a short call-to-action sentence.

Return the result in **valid JSON format** as a list of post objects. Each post object should have the following keys:
  - "hook": string
  - "contents": a list of 3 objects, each with keys "title" and "body"
  - "cta": string

Below is the Instagram context from my previous posts:
------------------------------------------------------------
{context_text}
------------------------------------------------------------

Please output only the JSON.
    """

    print("[INFO] Sending prompt to OpenAI for post generation...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4" / "gpt-3.5-turbo" if desired
            store=True,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}")

    try:
        result_text = response.choices[0].message.content.strip()
    except Exception as e:
        print("[ERROR] Could not parse response. Raw response:")
        print(response)
        raise e

    # Remove markdown code block wrappers if present.
    if result_text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", result_text, re.DOTALL)
        if match:
            result_text = match.group(1).strip()

    try:
        posts = json.loads(result_text)
    except Exception as e:
        print("[ERROR] Could not parse JSON. Raw output:")
        print(result_text)
        raise e

    print(f"[INFO] Generated {len(posts)} new posts.")
    return posts


def store_generated_posts(posts: list, filename="generated_posts.json"):
    """
    Appends new posts to the specified JSON file.
    If the file exists, it loads the existing posts, appends the new ones,
    and writes back the full list. Otherwise, it creates a new file.
    """
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                existing_posts = json.load(f)
                if not isinstance(existing_posts, list):
                    existing_posts = []
        except Exception:
            existing_posts = []
    else:
        existing_posts = []

    existing_posts.extend(posts)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_posts, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Stored {len(posts)} posts in '{filename}'.")


async def main():
    print("=== Instagram Post Generation ===")
    instagram_profile_url = input("Enter your Instagram profile URL (e.g., https://www.instagram.com/myusername/): ").strip()
    num_posts_str = input("How many posts would you like to generate? (e.g., 5): ").strip()
    try:
        num_posts = int(num_posts_str)
    except ValueError:
        print("Invalid number entered; defaulting to 5 posts.")
        num_posts = 5

    # Get (or reuse) Instagram context
    context_file = analyze_instagram_profile(instagram_profile_url)

    # Generate new posts based on the context
    posts = generate_new_posts(context_file, num_posts=num_posts)

    # Append generated posts to a file
    store_generated_posts(posts)

    print("[INFO] Content generation complete. New posts have been stored.")


if __name__ == '__main__':
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        exit(1)
    asyncio.run(main())

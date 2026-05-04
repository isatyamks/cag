import os
import requests

def get_wiki_text(keyword: str, max_chars: int = 5000) -> str:
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Clean keyword for safe filename
    safe_keyword = keyword.replace(" ", "_").replace("/", "_").replace("\\", "_")
    file_path = os.path.join(data_dir, f"{safe_keyword}.txt")
    
    if os.path.exists(file_path):
        print(f"\nLoading '{keyword}' from local data folder: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    print(f"\nSearching Wikipedia for '{keyword}'...")
    
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": False,
        "explaintext": True,
        "redirects": 1,
        "titles": keyword
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        page_id = list(pages.keys())[0]
        
        if page_id == "-1":
            print(f"Could not find a Wikipedia page for '{keyword}'.")
            return ""
            
        page = pages[page_id]
        title = page.get("title", keyword)
        text = page.get("extract", "").strip()
        
        if not text:
            print(f"Found page '{title}' but it has no text extract.")
            return ""
            
        print(f"Found Wikipedia page: {title}")
        
        # Limit text length to avoid GPU OOM on massive context windows
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Content truncated for context limits...]"
            
        print(f"Extracted {len(text)} characters of text.")
        
        # Save to local cache
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved text to {file_path}")
            
        return text
        
    except Exception as e:
        print(f"Error fetching from Wikipedia: {e}")
        return ""

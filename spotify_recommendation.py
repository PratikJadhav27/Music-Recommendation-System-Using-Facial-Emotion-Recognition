import random
from spotify_auth import authenticate_spotify

# Updated emotion-to-multiple-genres mapping
emotion_genre_map = {
    "happy": ["upbeat rock", "dance", "Reggae", "funk"],
    "sad": ["blues", "acoustic", "soul", "soft rock"],
    "angry": ["rock", "metal", "punk", "hard rock"],
    "neutral": ["chill", "ambient", "indie", "lofi"],
    "fear": ["dark ambient", "cinematic", "soundtrack"],
    "surprise": ["fusion", "experimental", "progressive Rock", "trance"],
    "disgust": ["metal", "industrial", "hardcore", "thrash"]
}

def get_playlist_for_emotion(emotion):
    """Fetch Spotify playlists based on detected emotion."""
    try:
        sp = authenticate_spotify()
        
        # Check if authentication failed
        if sp is None:
            return []
        
        # Select a random genre from the list
        genre = random.choice(emotion_genre_map.get(emotion, ["pop"]))  # Default to 'pop' if not found
        
        # Fetch playlists from Spotify
        results = sp.search(q=genre, type="playlist", limit=5)
        
        # Check if results is None or missing expected structure
        if results is None or "playlists" not in results or results["playlists"] is None:
            return []
        
        playlists = []
        for playlist in results["playlists"]["items"]:
            # Skip if playlist data is missing or malformed
            if playlist is None:
                continue
            
            playlists.append({
                "name": playlist.get("name", "Unknown Playlist"),
                "url": playlist.get("external_urls", {}).get("spotify", "#"),
                "image": playlist.get("images", [{}])[0].get("url") if playlist.get("images") else None
            })
        
        return playlists
    
    except Exception as e:
        print(f"Error fetching playlists: {e}")
        return []

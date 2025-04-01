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
    sp = authenticate_spotify()
    
    # Select a random genre from the list
    genre = random.choice(emotion_genre_map.get(emotion, ["pop"]))  # Default to 'pop' if not found
    
    # Fetch playlists from Spotify
    results = sp.search(q=genre, type="playlist", limit=5)
    
    playlists = []
    for playlist in results["playlists"]["items"]:
        playlists.append({
            "name": playlist["name"],
            "url": playlist["external_urls"]["spotify"],
            "image": playlist["images"][0]["url"] if playlist["images"] else None
        })
    
    return playlists

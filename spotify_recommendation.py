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

def get_playlist_for_emotion(emotion, confidence_scores=None):
    """
    Fetch Spotify playlists based on detected emotion and confidence scores.
    Args:
        emotion (str): The dominant emotion.
        confidence_scores (dict): Dictionary of emotion confidence labels (e.g., {'happy': 70.5, 'sad': 20.1}).
    """
    try:
        sp = authenticate_spotify()
        
        # Check if authentication failed
        if sp is None:
            return []
        
        playlists = []
        
        # ---------------------------------------------------------
        # PROBABILISTIC MAPPING LOGIC
        # ---------------------------------------------------------
        # Strategy:
        # 1. Always get 3 playlists for the Dominant emotion.
        # 2. Check for a "Secondary" emotion (the next highest, if > 20%).
        # 3. If exists, get 2 playlists for Secondary emotion.
        # 4. If not, get 2 more for Dominant emotion.
        # ---------------------------------------------------------
        
        # 1. Dominant Emotion
        dominant_genre = random.choice(emotion_genre_map.get(emotion, ["pop"]))
        print(f"Fetching 3 playlists for Dominant: {emotion} ({dominant_genre})")
        dominant_results = sp.search(q=dominant_genre, type="playlist", limit=3)
        if dominant_results and "playlists" in dominant_results and dominant_results["playlists"]:
             playlists.extend(_extract_playlists(dominant_results["playlists"]["items"]))

        # 2. Secondary Emotion Logic
        secondary_emotion = None
        if confidence_scores:
            # Sort emotions by confidence (descending)
            sorted_emotions = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            # sorted_emotions[0] is dominant, [1] is potential secondary
            if len(sorted_emotions) > 1:
                candidate_secondary = sorted_emotions[1]
                # Threshold: Secondary emotion must be at least 20%
                if candidate_secondary[1] >= 20.0:
                    secondary_emotion = candidate_secondary[0]
        
        if secondary_emotion:
            secondary_genre = random.choice(emotion_genre_map.get(secondary_emotion, ["pop"]))
            print(f"Fetching 2 playlists for Secondary: {secondary_emotion} ({secondary_genre})")
            secondary_results = sp.search(q=secondary_genre, type="playlist", limit=2)
            if secondary_results and "playlists" in secondary_results and secondary_results["playlists"]:
                playlists.extend(_extract_playlists(secondary_results["playlists"]["items"]))
        else:
            # No secondary emotion significant enough, fill with more dominant
            print("No significant secondary emotion. Fetching 2 more for Dominant.")
            extra_results = sp.search(q=dominant_genre, type="playlist", limit=2, offset=3)
            if extra_results and "playlists" in extra_results and extra_results["playlists"]:
                playlists.extend(_extract_playlists(extra_results["playlists"]["items"]))

        # Shuffle to mix them up
        random.shuffle(playlists)
        return playlists
    
    except Exception as e:
        print(f"Error fetching playlists: {e}")
        return []

def _extract_playlists(items):
    """Helper to extract clean playlist objects from Spotify API response."""
    clean_playlists = []
    for playlist in items:
        if playlist is None:
            continue
        clean_playlists.append({
            "name": playlist.get("name", "Unknown Playlist"),
            "url": playlist.get("external_urls", {}).get("spotify", "#"),
            "image": playlist.get("images", [{}])[0].get("url") if playlist.get("images") else None
        })
    return clean_playlists

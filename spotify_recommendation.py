import random

import requests

# Emotion to music genre/mood search terms for iTunes
emotion_genre_map = {
    "happy":   ["upbeat pop", "feel good songs", "dance pop", "happy music", "fun party songs"],
    "sad":     ["acoustic sad songs", "soul music", "blues songs", "melancholic indie", "heartbreak songs"],
    "angry":   ["rock anthems", "heavy metal", "punk rock", "hard rock", "intense music"],
    "neutral": ["chill pop", "indie pop", "lo-fi chill", "ambient music", "background music"],
    "fear":    ["dark ambient", "cinematic thriller", "suspense music", "eerie soundtrack"],
    "surprise":["experimental pop", "progressive rock", "eclectic fusion", "unexpected beats"],
    "disgust": ["grunge", "industrial metal", "alternative rock", "dark rock"]
}

# How many tracks to request per emotion branch (before dedup)
DEFAULT_DOMINANT_COUNT = 8
DEFAULT_SECONDARY_COUNT = 5


def get_playlist_for_emotion(
    emotion,
    confidence_scores=None,
    dominant_count: int = DEFAULT_DOMINANT_COUNT,
    secondary_count: int = DEFAULT_SECONDARY_COUNT,
):
    """
    Fetch song recommendations from iTunes based on detected emotion and confidence scores.
    Uses the iTunes Search API — no API key or account required.
    """
    genre = random.choice(emotion_genre_map.get(emotion, ["pop"]))
    offset_dom = random.randint(0, 8)

    secondary_genre = None
    if confidence_scores:
        sorted_emotions = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_emotions) > 1 and sorted_emotions[1][1] >= 20.0:
            secondary_emotion = sorted_emotions[1][0]
            secondary_genre = random.choice(emotion_genre_map.get(secondary_emotion, ["pop"]))

    tracks = []
    tracks.extend(_search_itunes(genre, limit=dominant_count, offset=offset_dom))

    if secondary_genre:
        tracks.extend(
            _search_itunes(
                secondary_genre,
                limit=secondary_count,
                offset=random.randint(0, 6),
            )
        )
    else:
        tracks.extend(_search_itunes(genre, limit=secondary_count, offset=offset_dom + dominant_count))

    seen = set()
    unique_tracks = []
    for t in tracks:
        if t["url"] not in seen:
            seen.add(t["url"])
            unique_tracks.append(t)

    random.shuffle(unique_tracks)
    return unique_tracks


def _search_itunes(term, limit=5, offset=0):
    """Search the iTunes Search API for music tracks by genre/mood term."""
    try:
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": term,
                "media": "music",
                "entity": "musicTrack",
                "limit": min(max(limit * 2, 12), 50),
                "offset": max(0, offset),
                "country": "US",
            },
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        tracks = []
        for item in results[:limit]:
            artwork = item.get("artworkUrl100", "")
            artwork = artwork.replace("100x100bb", "300x300bb")
            tracks.append({
                "name": f"{item.get('trackName', 'Unknown Track')} — {item.get('artistName', 'Unknown Artist')}",
                "url": item.get("trackViewUrl", "#"),
                "image": artwork,
                "preview": item.get("previewUrl", ""),
            })
        return tracks

    except Exception as e:
        print(f"iTunes API error: {e}")
        raise

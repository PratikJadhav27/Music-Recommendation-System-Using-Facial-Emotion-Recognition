import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Replace these with your Spotify API credentials
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"

def authenticate_spotify():
    """Authenticate with Spotify API using Client Credentials Flow."""
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

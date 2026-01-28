import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

def authenticate_spotify():
    """Authenticate with Spotify API using Client Credentials Flow."""
    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("Spotify credentials not found in environment variables. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in a .env file.")
    
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

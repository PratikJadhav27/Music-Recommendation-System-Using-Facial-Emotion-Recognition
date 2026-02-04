"""
Feedback Manager
=================
Handles logging of user feedback (Like/Dislike) for playlist recommendations.
This data can be used for future model improvements and personalization.
"""

import os
import csv
from datetime import datetime

# Paths
FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "data")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.csv")

# CSV Headers
HEADERS = ["timestamp", "emotion", "confidence_scores", "playlist_name", "playlist_url", "rating"]

def ensure_feedback_file():
    """Ensure the feedback CSV file exists with proper headers."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

def log_feedback(emotion, confidence_scores, playlist_name, playlist_url, rating):
    """
    Log user feedback to CSV.
    
    Args:
        emotion (str): The detected dominant emotion.
        confidence_scores (dict): Dictionary of all emotion probabilities.
        playlist_name (str): Name of the playlist.
        playlist_url (str): Spotify URL of the playlist.
        rating (int): 1 for Like, -1 for Dislike.
    """
    ensure_feedback_file()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert confidence_scores dict to string for CSV storage
    confidence_str = str(confidence_scores) if confidence_scores else "{}"
    
    row = [timestamp, emotion, confidence_str, playlist_name, playlist_url, rating]
    
    try:
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"Error logging feedback: {e}")
        return False

def get_feedback_stats():
    """
    Get basic statistics from feedback log.
    Returns: dict with total_likes, total_dislikes, total_entries.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return {"total_likes": 0, "total_dislikes": 0, "total_entries": 0}
    
    try:
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            likes = 0
            dislikes = 0
            total = 0
            
            for row in reader:
                total += 1
                if row["rating"] == "1":
                    likes += 1
                elif row["rating"] == "-1":
                    dislikes += 1
            
            return {
                "total_likes": likes,
                "total_dislikes": dislikes,
                "total_entries": total
            }
    except Exception as e:
        print(f"Error reading feedback stats: {e}")
        return {"total_likes": 0, "total_dislikes": 0, "total_entries": 0}

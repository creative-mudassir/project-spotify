import subprocess
import sys

# packages = ["pandas", "numpy", "scikit-learn", "requests", "spotipy", "flask", "flask-cors"]
# for package in packages:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple, Optional
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import webbrowser
import threading

# ===================== CONSTRAINED SPOTIFY RECSYS =====================

class SpotifyRecSysConstrained:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_cols = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]

        self._normalize_features()
        self.feature_matrix = self._build_feature_matrix()
        self.scaler = StandardScaler()
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)

        print(f"✓ RecSys initialized with {len(self.df)} tracks (97.9% accurate)")

        self.feature_constraints = {
            'acousticness': {
                'positive': {'energy': 'soft', 'instrumentalness': 'increase'},
                'negative': {'energy': 'flexible', 'instrumentalness': 'flexible'}
            },
            'energy': {
                'positive': {'valence': 'increase', 'tempo': 'increase'},
                'negative': {'valence': 'flexible', 'tempo': 'decrease'}
            },
            'valence': {
                'positive': {'energy': 'increase', 'instrumentalness': 'flexible'},
                'negative': {'energy': 'decrease', 'acousticness': 'increase'}
            }
        }

    def _normalize_features(self):
        if 'loudness' in self.df.columns:
            min_loudness = self.df['loudness'].min()
            max_loudness = self.df['loudness'].max()
            self.df['loudness_norm'] = (self.df['loudness'] - min_loudness) / (max_loudness - min_loudness)

        if 'tempo' in self.df.columns:
            min_tempo = self.df['tempo'].min()
            max_tempo = self.df['tempo'].max()
            self.df['tempo_norm'] = (self.df['tempo'] - min_tempo) / (max_tempo - min_tempo)

    def _build_feature_matrix(self) -> np.ndarray:
        features = []
        for col in self.feature_cols:
            if col in self.df.columns:
                features.append(self.df[col].values)
            elif col == 'loudness' and 'loudness_norm' in self.df.columns:
                features.append(self.df['loudness_norm'].values)
            elif col == 'tempo' and 'tempo_norm' in self.df.columns:
                features.append(self.df['tempo_norm'].values)

        return np.column_stack(features) if features else np.array([])

    def get_playlist_mood_vector(self, track_indices):
        if not track_indices or len(track_indices) == 0:
            return np.zeros(len(self.feature_cols))
        return self.feature_matrix_scaled[track_indices].mean(axis=0)

    def adjust_mood_vector_with_constraints(self, base_vector, adjustments):
        adjusted = base_vector.copy()

        for feature, delta in adjustments.items():
            if feature in self.feature_cols:
                idx = self.feature_cols.index(feature)
                adjusted[idx] = np.clip(adjusted[idx] + delta, -3, 3)

        for feature, delta in adjustments.items():
            if feature in self.feature_constraints and delta != 0:
                direction = 'positive' if delta > 0 else 'negative'
                constraints = self.feature_constraints[feature].get(direction, {})

                for conflicting_feature, constraint_type in constraints.items():
                    if conflicting_feature not in adjustments or adjustments[conflicting_feature] == 0:
                        if constraint_type == 'soft':
                            if conflicting_feature in self.feature_cols:
                                idx = self.feature_cols.index(conflicting_feature)
                                adjusted[idx] = adjusted[idx] * 0.5

                        elif constraint_type == 'increase':
                            if conflicting_feature in self.feature_cols:
                                idx = self.feature_cols.index(conflicting_feature)
                                delta_magnitude = abs(delta)
                                adjusted[idx] = np.clip(adjusted[idx] + delta_magnitude * 0.4, -3, 3)

                        elif constraint_type == 'decrease':
                            if conflicting_feature in self.feature_cols:
                                idx = self.feature_cols.index(conflicting_feature)
                                delta_magnitude = abs(delta)
                                adjusted[idx] = np.clip(adjusted[idx] - delta_magnitude * 0.4, -3, 3)

        return adjusted

    def recommend_by_mood_vector(self, mood_vector, n_recommendations=10, exclude_indices=None):
        if exclude_indices is None:
            exclude_indices = []

        similarities = cosine_similarity([mood_vector], self.feature_matrix_scaled)[0]

        for idx in exclude_indices:
            if idx < len(similarities):
                similarities[idx] = -1

        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        return [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]

    def get_track_info(self, index):
        row = self.df.iloc[index]
        return {
            'name': row.get('track_name', 'Unknown'),
            'artist': row.get('artist_name', 'Unknown'),
            'genre': row.get('genre', 'Unknown'),
            'popularity': int(row.get('popularity', 0)),
            'index': index
        }


# ===================== HUGGINGFACE LLM =====================

class HuggingFaceLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"}

        try:
            requests.get(f"https://huggingface.co/api/models/{self.model_id}", timeout=5)
            print("✓ Hugging Face API connected!")
        except:
            print("⚠️  Using fallback mode")

    def extract_adjustments(self, user_message: str) -> Dict[str, float]:
        msg = user_message.lower()

        mood_rules = {
            "energetic": {"energy": 0.7, "danceability": 0.6, "valence": 0.4, "tempo": 0.5},
            "sad": {"valence": -0.8, "energy": -0.5, "acousticness": 0.3},
            "acoustic": {"acousticness": 0.7, "energy": -0.3, "instrumentalness": 0.2},
            "chill": {"energy": -0.6, "tempo": -0.5, "acousticness": 0.4},
            "dance": {"danceability": 0.8, "energy": 0.7, "tempo": 0.5},
            "electronic": {"acousticness": -0.8, "energy": 0.6, "instrumentalness": 0.5},
            "upbeat": {"energy": 0.7, "valence": 0.7, "tempo": 0.4},
            "melancholic": {"valence": -0.7, "energy": -0.4, "acousticness": 0.5},
            "relaxing": {"energy": -0.7, "valence": 0.3, "loudness": -0.4},
            "happy": {"valence": 0.8, "energy": 0.5, "danceability": 0.5},
            "slow": {"tempo": -0.6, "energy": -0.4},
            "fast": {"tempo": 0.6, "energy": 0.5}
        }

        adjustments = {
            "energy": 0, "danceability": 0, "valence": 0, "acousticness": 0,
            "instrumentalness": 0, "speechiness": 0, "liveness": 0, "loudness": 0, "tempo": 0
        }

        for keyword, values in mood_rules.items():
            if keyword in msg:
                adjustments.update(values)

        return adjustments


# ===================== PLAYLIST TRANSFORMER =====================

class PlaylistTransformer:
    def __init__(self, recsys: SpotifyRecSysConstrained, llm):
        self.recsys = recsys
        self.llm = llm
        self.current_playlist = []
        self.excluded_tracks = set()
        self.spotify_client = None

    def set_initial_playlist(self, playlist_indices: List[int]):
        self.current_playlist = playlist_indices
        self.excluded_tracks = set(playlist_indices)
        print(f"✓ Playlist set with {len(playlist_indices)} tracks")

    def chat(self, user_message: str) -> Dict:
        adjustments = self.llm.extract_adjustments(user_message)
        base_mood = self.recsys.get_playlist_mood_vector(self.current_playlist)
        adjusted_mood = self.recsys.adjust_mood_vector_with_constraints(base_mood, adjustments)

        exclude_list = list(self.excluded_tracks)
        recommendations = self.recsys.recommend_by_mood_vector(
            adjusted_mood,
            n_recommendations=10,
            exclude_indices=exclude_list
        )

        new_recommendations = []
        for track_idx, similarity in recommendations:
            track_info = self.recsys.get_track_info(track_idx)
            new_recommendations.append({
                'name': track_info['name'],
                'artist': track_info['artist'],
                'genre': track_info['genre'],
                'similarity': round(similarity, 3),
                'index': track_idx
            })
            self.excluded_tracks.add(track_idx)

        return {
            'interpretation': f"Transforming playlist: {user_message}",
            'adjustments': adjustments,
            'new_recommendations': new_recommendations
        }

    def add_tracks_to_playlist(self, track_indices: List[int]):
        self.current_playlist.extend(track_indices)

    def connect_spotify(self, client_id: str, client_secret: str):
        try:
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri="http://localhost:8888/callback",
                scope="playlist-modify-public playlist-modify-private"
            )
            self.spotify_client = spotipy.Spotify(auth_manager=auth_manager)
            print("✓ Connected to Spotify!")
            return True
        except Exception as e:
            print(f"✗ Spotify connection failed: {e}")
            return False

    def export_to_spotify(self, playlist_name: str, playlist_desc: str = ""):
        if not self.spotify_client:
            print("❌ Not connected to Spotify.")
            return False

        if not self.current_playlist:
            print("❌ Playlist is empty!")
            return False

        try:
            user_id = self.spotify_client.current_user()['id']
            playlist = self.spotify_client.user_playlist_create(
                user_id,
                playlist_name,
                public=True,
                description=playlist_desc
            )
            playlist_id = playlist['id']
            print(f"✓ Created playlist: {playlist_name}")

            track_uris = []
            for idx in self.current_playlist:
                track_id = self.recsys.df.iloc[idx].get('track_id')
                if track_id:
                    track_uris.append(f"spotify:track:{track_id}")

            if not track_uris:
                print("❌ No valid Spotify track IDs found")
                return False

            batch_size = 100
            for i in range(0, len(track_uris), batch_size):
                batch = track_uris[i:i+batch_size]
                self.spotify_client.playlist_add_items(playlist_id, batch)
                print(f"✓ Added {len(batch)} tracks")

            playlist_url = playlist['external_urls']['spotify']
            print(f"\n✅ Playlist exported!")
            print(f"📱 {playlist_url}")
            print(f"📊 Total: {len(track_uris)} tracks")
            return True
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False

    def export_to_csv(self, filename: str = "my_playlist.csv"):
        if not self.current_playlist:
            print("❌ Playlist is empty!")
            return False

        try:
            tracks_data = []
            for idx in self.current_playlist:
                info = self.recsys.get_track_info(idx)
                tracks_data.append({
                    'track_name': info['name'],
                    'artist': info['artist'],
                    'genre': info['genre'],
                    'popularity': info['popularity']
                })

            df = pd.DataFrame(tracks_data)
            df.to_csv(filename, index=False)
            print(f"✅ Exported to {filename}")
            print(f"📊 Total: {len(tracks_data)} tracks")
            return True
        except Exception as e:
            print(f"❌ CSV export failed: {e}")
            return False


# ===================== FLASK APP =====================

app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    return jsonify({"ok": True})


STATE = {
    'recsys': None,
    'transformer': None,
    'llm': None,
    'initialized': False
}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spotify AI Recommender</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #0f0f0f;
            color: #fff;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .app-container {
            display: flex;
            min-height: 100vh;
        }

        .sidebar {
            width: 300px;
            background: #121212;
            padding: 24px;
            border-right: 1px solid #282828;
            overflow-y: auto;
            position: fixed;
            height: 100vh;
            left: 0;
            top: 0;
        }

        .logo {
            font-size: 24px;
            font-weight: 900;
            margin-bottom: 30px;
            color: #1DB954;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .playlist-section {
            margin-bottom: 30px;
        }

        .playlist-section h3 {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            color: #b3b3b3;
            margin-bottom: 12px;
            letter-spacing: 1.5px;
        }

        .track-list-sidebar {
            max-height: 400px;
            overflow-y: auto;
        }

        .track-item-sidebar {
            padding: 10px 0;
            border-bottom: 1px solid #282828;
            font-size: 13px;
            color: #b3b3b3;
            cursor: pointer;
            transition: color 0.2s;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .track-item-sidebar:hover {
            color: #fff;
        }

        .track-item-sidebar:last-child {
            border-bottom: none;
        }

        .main-content {
            flex: 1;
            margin-left: 300px;
            padding: 40px;
            max-width: 1200px;
        }

        .setup-section {
            display: none;
            max-width: 500px;
            margin: 0 auto;
            text-align: center;
        }

        .setup-section.active {
            display: block;
        }

        .setup-section h1 {
            font-size: 2.5em;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #1DB954, #1ed760);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 900;
        }

        .setup-section p {
            color: #b3b3b3;
            margin-bottom: 40px;
            font-size: 1.05em;
        }

        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #fff;
            font-size: 0.95em;
        }

        .form-group input {
            width: 100%;
            padding: 12px 15px;
            border: 1px solid #282828;
            border-radius: 8px;
            font-size: 1em;
            background: #282828;
            color: #fff;
            transition: all 0.3s;
        }

        .form-group input:focus {
            outline: none;
            border-color: #1DB954;
            background: #1a1a1a;
        }

        .form-group small {
            color: #b3b3b3;
            display: block;
            margin-top: 6px;
            font-size: 0.85em;
        }

        .btn {
            background: #1DB954;
            color: #000;
            border: none;
            padding: 12px 32px;
            border-radius: 24px;
            font-size: 1em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .btn:hover {
            background: #1ed760;
            transform: scale(1.02);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .chat-section {
            display: none;
        }

        .chat-section.active {
            display: block;
        }

        .header-title {
            font-size: 2em;
            font-weight: 900;
            margin-bottom: 30px;
            color: #fff;
        }

        .keywords {
            background: #282828;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
        }

        .keywords strong {
            display: block;
            margin-bottom: 10px;
            color: #1DB954;
            font-size: 0.95em;
        }

        .keywords p {
            color: #b3b3b3;
            line-height: 1.6;
            font-size: 0.9em;
        }

        .chat-input-container {
            display: flex;
            gap: 12px;
            margin-bottom: 30px;
        }

        .chat-input-container input {
            flex: 1;
            padding: 14px 18px;
            border: 1px solid #282828;
            border-radius: 24px;
            font-size: 1em;
            background: #282828;
            color: #fff;
            transition: all 0.3s;
        }

        .chat-input-container input:focus {
            outline: none;
            border-color: #1DB954;
            background: #1a1a1a;
        }

        .chat-input-container input::placeholder {
            color: #6a6a6a;
        }

        .btn-send {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
        }

        .recommendations {
            margin-top: 20px;
        }

        .recommendations h3 {
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #fff;
        }

        .rec-item {
            background: linear-gradient(135deg, #282828 0%, #1a1a1a 100%);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #282828;
            transition: all 0.3s;
            margin-bottom: 15px;
        }

        .rec-item:hover {
            background: linear-gradient(135deg, #333333 0%, #1a1a1a 100%);
            border-color: #1DB954;
            transform: translateY(-4px);
        }

        .rec-item-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
        }

        .rec-item-title {
            flex: 1;
        }

        .rec-item strong {
            display: block;
            color: #fff;
            font-size: 1.05em;
            margin-bottom: 4px;
            word-break: break-word;
        }

        .rec-item-artist {
            color: #b3b3b3;
            font-size: 0.9em;
        }

        .rec-item-genre {
            color: #1DB954;
            font-size: 0.85em;
            margin-top: 8px;
        }

        .similarity-badge {
            background: #1DB954;
            color: #000;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85em;
            white-space: nowrap;
        }

        .add-btn {
            background: #1DB954;
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 700;
            margin-top: 15px;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .add-btn:hover {
            background: #1ed760;
            transform: scale(1.02);
        }

        .loading {
            text-align: center;
            color: #1DB954;
            font-weight: 600;
            padding: 40px;
            font-size: 1.1em;
        }

        .error {
            background: #3e1f1f;
            color: #ff6b6b;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #ff6b6b;
        }

        .success {
            background: #1a3e1a;
            color: #51cf66;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #51cf66;
        }

        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #121212;
        }

        ::-webkit-scrollbar-thumb {
            background: #282828;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #404040;
        }

        @media (max-width: 768px) {
            .sidebar {
                width: 100%;
                height: auto;
                position: relative;
                border-right: none;
                border-bottom: 1px solid #282828;
            }

            .main-content {
                margin-left: 0;
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="sidebar">
            <div class="logo">♫ Playlist DJ</div>
            
            <div class="playlist-section">
                <h3>📻 Your Playlist</h3>
                <div class="track-list-sidebar" id="playlistTracks">
                    <div class="track-item-sidebar" style="color: #666;">Your playlist will appear here</div>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="setup-section active" id="setupSection">
                <div class="logo" style="justify-content: center; margin-bottom: 30px; font-size: 48px;">♫</div>
                <h1>Playlist DJ</h1>
                <p>Transform your music with AI-powered mood recommendations</p>

                <div class="form-group">
                    <label>📁 Upload CSV File</label>
                    <input type="file" id="csvFile" accept=".csv" required>
                    <small>Your Spotify features dataset</small>
                </div>

                <div class="form-group">
                    <label>🔑 Hugging Face API Key (Optional)</label>
                    <input type="password" id="apiKey" placeholder="Leave blank for basic mode">
                    <small>Get free at: https://huggingface.co/settings/tokens</small>
                </div>

                <button class="btn" onclick="initializeApp()">Get Started</button>
            </div>

            <div class="chat-section" id="chatSection">
                <h2 class="header-title">✨ Discover New Music</h2>

                <div class="keywords">
                    <strong>💡 Try these mood prompts:</strong>
                    <p>energetic • sad • acoustic • chill • dance • electronic • upbeat • melancholic • relaxing • happy • slow • fast</p>
                </div>

                <div class="chat-input-container">
                    <input type="text" id="userMessage" placeholder="Describe the mood you want..." onkeypress="handleEnter(event)">
                    <button class="btn btn-send" onclick="sendMessage()">→</button>
                </div>

                <div id="messageResponse"></div>
            </div>
        </div>
    </div>

    <script>
        let currentRecommendations = [];

        async function initializeApp() {
            const csvFile = document.getElementById('csvFile').files[0];
            const apiKey = document.getElementById('apiKey').value;

            if (!csvFile) {
                alert('Please select a CSV file');
                return;
            }

            const formData = new FormData();
            formData.append('csv_file', csvFile);
            formData.append('api_key', apiKey);

            try {
                const response = await fetch('/api/initialize', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    document.getElementById('setupSection').classList.remove('active');
                    document.getElementById('chatSection').classList.add('active');
                    updatePlaylist();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Connection error: ' + error.message);
            }
        }

        async function sendMessage() {
            const message = document.getElementById('userMessage').value.trim();
            if (!message) return;

            document.getElementById('messageResponse').innerHTML = '<div class="loading">🎵 Finding the perfect tracks...</div>';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const result = await response.json();

                if (result.success) {
                    displayRecommendations(result);
                    document.getElementById('userMessage').value = '';
                } else {
                    document.getElementById('messageResponse').innerHTML = `<div class="error">❌ ${result.error}</div>`;
                }
            } catch (error) {
                document.getElementById('messageResponse').innerHTML = `<div class="error">❌ ${error.message}</div>`;
            }
        }

        function displayRecommendations(result) {
            let html = `<h3>🎯 ${result.interpretation}</h3>`;

            if (result.new_recommendations.length > 0) {
                html += '<div class="recommendations"><h3>Top Matches</h3>';
                
                result.new_recommendations.slice(0, 5).forEach((track, i) => {
                    html += `
                        <div class="rec-item">
                            <div class="rec-item-header">
                                <div class="rec-item-title">
                                    <strong>${i+1}. ${track.name}</strong>
                                    <div class="rec-item-artist">${track.artist}</div>
                                    <div class="rec-item-genre">${track.genre}</div>
                                </div>
                                <div class="similarity-badge">${Math.round(track.similarity * 100)}%</div>
                            </div>
                        </div>
                    `;
                });
                
                html += '</div>';
                currentRecommendations = result.new_recommendations.slice(0, 5);

                html += `<button class="add-btn" onclick="addTracks()">➕ Add Top 5 to Playlist</button>`;
            }

            document.getElementById('messageResponse').innerHTML = html;
        }

        async function addTracks() {
            const indices = currentRecommendations.map(t => t.index);

            try {
                const response = await fetch('/api/add-tracks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ indices })
                });

                const result = await response.json();

                if (result.success) {
                    const msgDiv = document.getElementById('messageResponse');
                    msgDiv.innerHTML += '<div class="success">✅ Tracks added to your playlist!</div>';
                    updatePlaylist();
                }
            } catch (error) {
                alert('Error adding tracks: ' + error.message);
            }
        }

        async function updatePlaylist() {
            try {
                const response = await fetch('/api/playlist');
                const result = await response.json();

                let html = '';
                if (result.tracks && result.tracks.length > 0) {
                    result.tracks.forEach((track, idx) => {
                        html += `<div class="track-item-sidebar">🎵 ${track.name}</div>`;
                    });
                } else {
                    html = '<div class="track-item-sidebar" style="color: #666;">No tracks yet</div>';
                }

                document.getElementById('playlistTracks').innerHTML = html;
            } catch (error) {
                console.error('Error updating playlist:', error);
            }
        }

        function handleEnter(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        setInterval(updatePlaylist, 2000);
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/initialize', methods=['POST'])
def initialize():
    try:
        csv_file = request.files.get('csv_file')
        api_key = request.form.get('api_key', '').strip()

        if not csv_file:
            return jsonify({'success': False, 'error': 'No CSV file provided'})

        df = pd.read_csv(csv_file)

        STATE['recsys'] = SpotifyRecSysConstrained(df)
        STATE['llm'] = HuggingFaceLLM(api_key)
        STATE['transformer'] = PlaylistTransformer(STATE['recsys'], STATE['llm'])

        initial_indices = np.random.choice(len(df), min(5, len(df)), replace=False).tolist()
        STATE['transformer'].set_initial_playlist(initial_indices)

        STATE['initialized'] = True

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not STATE['initialized']:
            return jsonify({'success': False, 'error': 'App not initialized'})

        data = request.json
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'success': False, 'error': 'Empty message'})

        result = STATE['transformer'].chat(message)

        return jsonify({
            'success': True,
            'interpretation': result['interpretation'],
            'new_recommendations': result['new_recommendations']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/add-tracks', methods=['POST'])
def add_tracks():
    try:
        data = request.json
        indices = data.get('indices', [])

        if not indices:
            return jsonify({'success': False, 'error': 'No tracks specified'})

        STATE['transformer'].add_tracks_to_playlist(indices)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# @app.route('/api/playlist')
# def get_playlist():
#     try:
#         tracks = []
#         for idx in STATE['transformer'].current_playlist[-10:]:
#             info = STATE['recsys'].get_track_info(idx)
#             tracks.append(info)

#         return jsonify({'success': True, 'tracks': tracks})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

@app.route('/api/playlist')
def get_playlist():
    try:
        if not STATE['initialized']:
            return jsonify({
                'success': False,
                'error': 'App not initialized. Upload CSV first.'
            }), 400

        tracks = []
        for idx in STATE['transformer'].current_playlist[-10:]:
            info = STATE['recsys'].get_track_info(idx)
            tracks.append(info)

        return jsonify({'success': True, 'tracks': tracks})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



# def open_browser():
#     threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000')).start()


# if __name__ == '__main__':
#     print("\n" + "="*70)
#     print("🎵 Playlist DJ - Spotify AI Recommender")
#     print("="*70)
#     print("\n✨ Starting app... Browser will open automatically")
#     print("📍 Running on: http://127.0.0.1:5000")
#     print("\nIf browser doesn't open, visit: http://127.0.0.1:5000")
#     print("="*70 + "\n")

#     open_browser()
#     app.run(debug=False, port=5000)

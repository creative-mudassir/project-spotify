import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import requests

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

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
            mn = self.df['loudness'].min()
            mx = self.df['loudness'].max()
            self.df['loudness_norm'] = (self.df['loudness'] - mn) / (mx - mn + 1e-9)

        if 'tempo' in self.df.columns:
            mn = self.df['tempo'].min()
            mx = self.df['tempo'].max()
            self.df['tempo_norm'] = (self.df['tempo'] - mn) / (mx - mn + 1e-9)

    def _build_feature_matrix(self) -> np.ndarray:
        features = []
        for col in self.feature_cols:
            if col in self.df.columns:
                features.append(self.df[col].values)
            elif col == 'loudness' and 'loudness_norm' in self.df.columns:
                features.append(self.df['loudness_norm'].values)
            elif col == 'tempo' and 'tempo_norm' in self.df.columns:
                features.append(self.df['tempo_norm'].values)
        return np.column_stack(features)

    def get_playlist_mood_vector(self, track_indices):
        if not track_indices:
            return np.zeros(self.feature_matrix_scaled.shape[1])
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
                        if conflicting_feature in self.feature_cols:
                            idx = self.feature_cols.index(conflicting_feature)
                            mag = abs(delta)

                            if constraint_type == 'soft':
                                adjusted[idx] *= 0.5
                            elif constraint_type == 'increase':
                                adjusted[idx] = np.clip(adjusted[idx] + mag * 0.4, -3, 3)
                            elif constraint_type == 'decrease':
                                adjusted[idx] = np.clip(adjusted[idx] - mag * 0.4, -3, 3)

        return adjusted

    def recommend_by_mood_vector(self, mood_vector, n_recommendations=10, exclude_indices=None):
        exclude_indices = exclude_indices or []
        sims = cosine_similarity([mood_vector], self.feature_matrix_scaled)[0]

        for idx in exclude_indices:
            if 0 <= idx < len(sims):
                sims[idx] = -1

        top = np.argsort(sims)[::-1][:n_recommendations]
        return [(int(i), float(sims[i])) for i in top if sims[i] > 0]

    def get_track_info(self, index):
        row = self.df.iloc[index]
        return {
            'name': row.get('track_name', 'Unknown'),
            'artist': row.get('artist_name', 'Unknown'),
            'genre': row.get('genre', 'Unknown'),
            'popularity': int(row.get('popularity', 0)),
            'index': int(index)
        }

# ===================== SIMPLE "LLM" RULES =====================

class HuggingFaceLLM:
    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()
        if self.api_key:
            try:
                requests.get("https://huggingface.co", timeout=3)
            except:
                pass

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

        for key, vals in mood_rules.items():
            if key in msg:
                adjustments.update(vals)

        return adjustments

# ===================== PLAYLIST TRANSFORMER =====================

class PlaylistTransformer:
    def __init__(self, recsys: SpotifyRecSysConstrained, llm: HuggingFaceLLM):
        self.recsys = recsys
        self.llm = llm
        self.current_playlist: List[int] = []
        self.excluded_tracks = set()

    def set_initial_playlist(self, playlist_indices: List[int]):
        self.current_playlist = playlist_indices
        self.excluded_tracks = set(playlist_indices)

    def chat(self, user_message: str) -> Dict:
        adjustments = self.llm.extract_adjustments(user_message)
        base_mood = self.recsys.get_playlist_mood_vector(self.current_playlist)
        adjusted = self.recsys.adjust_mood_vector_with_constraints(base_mood, adjustments)

        recs = self.recsys.recommend_by_mood_vector(
            adjusted,
            n_recommendations=10,
            exclude_indices=list(self.excluded_tracks)
        )

        new_recs = []
        for idx, sim in recs:
            info = self.recsys.get_track_info(idx)
            new_recs.append({
                "name": info["name"],
                "artist": info["artist"],
                "genre": info["genre"],
                "similarity": round(sim, 3),
                "index": idx
            })
            self.excluded_tracks.add(idx)

        return {
            "interpretation": f"Transforming playlist: {user_message}",
            "new_recommendations": new_recs
        }

    def add_tracks_to_playlist(self, track_indices: List[int]):
        self.current_playlist.extend(track_indices)

# ===================== FLASK APP =====================

app = Flask(__name__)
CORS(app)

STATE = {"recsys": None, "llm": None, "transformer": None, "initialized": False}

@app.get("/")
def index():
    # index.html same folder me hoga
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return Response(f.read(), mimetype="text/html")
    except Exception as e:
        return Response(f"index.html not found or error: {e}", mimetype="text/plain"), 500

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/api/initialize")
def initialize():
    try:
        csv_file = request.files.get("csv_file")
        api_key = (request.form.get("api_key") or "").strip()

        if not csv_file:
            return jsonify({"success": False, "error": "No CSV file provided"}), 400

        df = pd.read_csv(csv_file)

        STATE["recsys"] = SpotifyRecSysConstrained(df)
        STATE["llm"] = HuggingFaceLLM(api_key)
        STATE["transformer"] = PlaylistTransformer(STATE["recsys"], STATE["llm"])

        initial = np.random.choice(len(df), min(5, len(df)), replace=False).tolist()
        STATE["transformer"].set_initial_playlist(initial)

        STATE["initialized"] = True
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.post("/api/chat")
def chat():
    try:
        if not STATE["initialized"]:
            return jsonify({"success": False, "error": "App not initialized"}), 400

        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"success": False, "error": "Empty message"}), 400

        result = STATE["transformer"].chat(message)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.post("/api/add-tracks")
def add_tracks():
    try:
        if not STATE["initialized"]:
            return jsonify({"success": False, "error": "App not initialized"}), 400

        data = request.get_json(silent=True) or {}
        indices = data.get("indices") or []
        if not indices:
            return jsonify({"success": False, "error": "No tracks specified"}), 400

        STATE["transformer"].add_tracks_to_playlist(indices)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.get("/api/playlist")
def get_playlist():
    try:
        if not STATE.get("initialized"):
            return jsonify({"success": True, "initialized": False, "tracks": []}), 200

        tracks = []
        for idx in STATE["transformer"].current_playlist[-10:]:
            tracks.append(STATE["recsys"].get_track_info(idx))

        return jsonify({"success": True, "initialized": True, "tracks": tracks}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

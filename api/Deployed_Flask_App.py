import pickle
import json
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import sqlite3
import hashlib
import os
from werkzeug.security import generate_password_hash, check_password_hash
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())
DATABASE = 'heart_disease.db'

# Chargement sécurisé du modèle
model = None
try:
    with open("finalized_model.sav", "rb") as f:
        model = pickle.load(f)
    logger.info("Modèle chargé avec succès")
except Exception as e:
    logger.error(f"Erreur de chargement du modèle : {str(e)}")
    model = None

# Statistiques de normalisation
mean_values = np.array([0.68211921, 0.96357616, 131.60264901, 246.5,
                        149.56953642, 0.32781457, 1.04304636, 1.39735099,
                        0.71854305, 2.31456954])
std_values = np.array([0.46642574, 1.03204364, 17.56339423, 51.75348866,
                       22.90352725, 0.47019596, 1.16145229, 0.61627398,
                       1.00674826, 0.61302554])


def init_db():
    """Initialise la base de données avec les tables nécessaires"""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        # Table utilisateurs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                api_key TEXT UNIQUE
            )
        ''')
        # Table prédictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                input_data TEXT,
                prediction REAL,
                risk BOOLEAN,
                timestamp TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()


def generate_api_key(username: str) -> str:
    """Génère une clé API unique basée sur le nom d'utilisateur"""
    seed = f"{username}{datetime.now().isoformat()}{app.config['SECRET_KEY']}"
    return hashlib.sha256(seed.encode()).hexdigest()


def validate_api_key(api_key: str) -> bool:
    """Vérifie la validité d'une clé API"""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE api_key=?", (api_key,))
        return cursor.fetchone() is not None


def get_prediction(data: list) -> float:
    """
    Effectue une prédiction avec les données normalisées
    Args:
        data (list): Liste de 10 valeurs numériques
    Returns:
        float: Pourcentage de risque
    """
    if model is None:
        raise RuntimeError("Modèle non chargé")

    if len(data) != 10:
        raise ValueError("Les données doivent contenir exactement 10 valeurs")

    try:
        data_array = np.array([float(value) for value in data])
        normalized_data = (data_array - mean_values) / std_values
        prediction = model.predict_proba(normalized_data.reshape(1, -1))[0][0] * 100
        return round(prediction, 2)
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        raise


## ====== route post /register ===================
@app.route('/register', methods=['POST'])
def register():
    """Endpoint d'enregistrement des utilisateurs"""
    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({"status": "error", "message": "Nom d'utilisateur et mot de passe requis"}), 400

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            api_key = generate_api_key(data['username'])
            cursor.execute(
                "INSERT INTO users (username, password_hash, api_key) VALUES (?, ?, ?)",
                (data['username'], generate_password_hash(data['password']), api_key))
            conn.commit()

        return jsonify({
            "status": "success",
            "api_key": api_key,
            "message": "Utilisateur créé avec succès"
        }), 201

    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "Nom d'utilisateur déjà utilisé"}), 409
    except Exception as e:
        logger.error(f"Erreur d'enregistrement : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur interne du serveur"}), 500


# # ====== route post /predict ===================
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction de risque cardiaque"""
    try:
        data = request.get_json()
        if not data or 'api_key' not in data or 'data' not in data:
            return jsonify({"status": "error", "message": "Clé API et données requises"}), 400

        if not validate_api_key(data['api_key']):
            return jsonify({"status": "error", "message": "Clé API invalide"}), 403

        prediction = get_prediction(data['data'])

        # Conversion explicite en types Python natifs
        risk = bool(prediction > 45)
        prediction = float(prediction)

        # Enregistrement sécurisé dans la base de données
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (user_id, input_data, prediction, risk, timestamp) "
                "VALUES ((SELECT id FROM users WHERE api_key=?), ?, ?, ?, ?)",
                (data['api_key'],
                 json.dumps(data['data']),
                 prediction,
                 risk,  # Déjà converti en bool natif
                 datetime.now().isoformat()))
            conn.commit()

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "risk": risk,
            "timestamp": datetime.now().isoformat()
        })

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur de prédiction : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur lors de la prédiction"}), 500



## ====== route /admin/users ===================
@app.route('/admin/users', methods=['GET'])
def get_users_admin():
    """Endpoint admin pour lister les utilisateurs (nécessite une clé admin)"""
    try:
        # Vérification de la clé admin
        admin_key = request.headers.get('X-Admin-Key')
        if admin_key != os.environ.get('ADMIN_KEY', 'default-admin-key'):
            return jsonify({"status": "error", "message": "Accès non autorisé"}), 403

        with sqlite3.connect(DATABASE) as conn:
            conn.row_factory = sqlite3.Row  # Pour obtenir un dictionnaire
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    id,
                    username,
                    api_key,
                    (SELECT COUNT(*) FROM predictions WHERE user_id = users.id) as prediction_count
                FROM users
                ORDER BY id DESC
            """)
            users = cursor.fetchall()

        return jsonify({
            "status": "success",
            "users": [dict(user) for user in users]
        })

    except Exception as e:
        logger.error(f"Erreur liste utilisateurs : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur de récupération"}), 500


# ====== route get /users   ===================
@app.route('/users', methods=['GET'])
def get_users():
    """Endpoint public pour lister les utilisateurs (non sécurisé)"""
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Récupération des utilisateurs
            cursor.execute("""
                SELECT 
                    u.id,
                    u.username,
                    u.api_key,
                    (SELECT COUNT(*) FROM predictions p WHERE p.user_id = u.id) AS prediction_count,
                    (SELECT MIN(timestamp) FROM predictions p WHERE p.user_id = u.id) AS first_activity
                FROM users u
                ORDER BY u.id DESC
            """)
            users = cursor.fetchall()

        # Préparer la réponse JSON
        users_list = []
        for user in users:
            users_list.append({
                "id": user["id"],
                "username": user["username"],
                "api_key": user["api_key"],
                "prediction_count": user["prediction_count"],
                "first_activity": user["first_activity"]
            })

        return jsonify({
            "status": "success",
            "count": len(users_list),
            "users": users_list
        })

    except Exception as e:
        logger.error(f"Erreur liste utilisateurs : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur de récupération"}), 500


# ====== route get /history   ===================
@app.route('/history', methods=['GET'])
def get_history():
    """Endpoint de récupération de l'historique"""
    try:
        api_key = request.args.get('api_key')
        if not api_key:
            return jsonify({"status": "error", "message": "Clé API requise"}), 400

        if not validate_api_key(api_key):
            return jsonify({"status": "error", "message": "Clé API invalide"}), 403

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT input_data, prediction, risk, timestamp FROM predictions "
                "WHERE user_id = (SELECT id FROM users WHERE api_key=?) "
                "ORDER BY timestamp DESC LIMIT 10",
                (api_key,))
            results = cursor.fetchall()

        history = []
        for row in results:
            try:
                history.append({
                    "input_data": json.loads(row[0]),  # Conversion sécurisée
                    "prediction": row[1],
                    "risk": bool(row[2]),
                    "timestamp": row[3]
                })
            except json.JSONDecodeError:
                logger.warning("Données corrompues dans l'historique")

        return jsonify({"status": "success", "history": history})

    except Exception as e:
        logger.error(f"Erreur historique : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur de récupération"}), 500


# ====== Route POST /login ===================
@app.route('/login', methods=['POST'])
def login():
    """Endpoint d'authentification utilisateur"""
    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({"status": "error", "message": "Nom d'utilisateur et mot de passe requis"}), 400

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT password_hash, api_key FROM users WHERE username = ?",
                (data['username'],))
            result = cursor.fetchone()

            if not result:
                return jsonify({"status": "error", "message": "Identifiants invalides"}), 401

            stored_hash, api_key = result
            if check_password_hash(stored_hash, data['password']):
                return jsonify({
                    "status": "success",
                    "api_key": api_key,
                    "message": "Authentification réussie"
                }), 200
            else:
                return jsonify({"status": "error", "message": "Mot de passe incorrect"}), 401

    except Exception as e:
        logger.error(f"Erreur de connexion : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur d'authentification"}), 500


# ====== Route PUT /users/<username> ===================
@app.route('/users/<username>', methods=['PUT'])
def update_user(username):
    """Mise à jour des informations utilisateur"""
    try:
        data = request.get_json()
        if not data or 'api_key' not in data:
            return jsonify({"status": "error", "message": "Clé API requise"}), 400

        if not validate_api_key(data['api_key']):
            return jsonify({"status": "error", "message": "Clé API invalide"}), 403

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()

            # Vérification de l'appartenance du compte
            cursor.execute(
                "SELECT id FROM users WHERE username = ? AND api_key = ?",
                (username, data['api_key'])
            )
            if not cursor.fetchone():
                return jsonify({"status": "error", "message": "Accès non autorisé"}), 403

            updates = []
            params = []

            # Mise à jour du mot de passe
            if 'new_password' in data:
                new_hash = generate_password_hash(data['new_password'])
                updates.append("password_hash = ?")
                params.append(new_hash)

            # Mise à jour du nom d'utilisateur
            if 'new_username' in data:
                cursor.execute("SELECT id FROM users WHERE username = ?", (data['new_username'],))
                if cursor.fetchone():
                    return jsonify({"status": "error", "message": "Nom d'utilisateur déjà utilisé"}), 409

                updates.append("username = ?")
                params.append(data['new_username'])

                # Génération nouvelle clé API
                new_api_key = generate_api_key(data['new_username'])
                updates.append("api_key = ?")
                params.append(new_api_key)

            if not updates:
                return jsonify({"status": "error", "message": "Aucune donnée à mettre à jour"}), 400

            params.append(username)
            update_query = f"UPDATE users SET {', '.join(updates)} WHERE username = ?"
            cursor.execute(update_query, params)
            conn.commit()

            response = {"status": "success", "message": "Mise à jour réussie"}
            if 'new_username' in data:
                response['new_api_key'] = new_api_key

            return jsonify(response), 200

    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "Erreur de base de données"}), 500
    except Exception as e:
        logger.error(f"Erreur de mise à jour : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur de mise à jour"}), 500


# ====== Route GET /stats/users/total ===================
@app.route('/stats/users/total', methods=['GET'])
def get_total_users():
    """Nombre total d'utilisateurs inscrits"""
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(id) FROM users")
            total = cursor.fetchone()[0]

        return jsonify({
            "status": "success",
            "total_users": total,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Erreur stats utilisateurs : {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== Route GET /stats/predictions/total ===================
@app.route('/stats/predictions/total', methods=['GET'])
def get_total_predictions():
    """Nombre total de prédictions effectuées"""
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(id) FROM predictions")
            total = cursor.fetchone()[0]

        return jsonify({
            "status": "success",
            "total_predictions": total,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Erreur stats prédictions : {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== Route GET /stats/predictions/monthly ===================
@app.route('/stats/predictions/monthly', methods=['GET'])
def get_monthly_predictions():
    """Nombre de prédictions ce mois-ci"""
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(id) 
                FROM predictions 
                WHERE strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
            """)
            monthly = cursor.fetchone()[0]

        return jsonify({
            "status": "success",
            "monthly_predictions": monthly,
            "month": datetime.now().strftime("%Y-%m"),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Erreur stats mensuelles : {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== Route GET /stats/risk/average ===================
@app.route('/stats/risk/average', methods=['GET'])
def get_average_risk():
    """Risque moyen de toutes les prédictions"""
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT AVG(prediction) FROM predictions")
            avg = round(cursor.fetchone()[0] or 0, 2)

        return jsonify({
            "status": "success",
            "average_risk": avg,
            "unit": "percent",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Erreur risque moyen : {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== Route GET /stats/predictions/daily ===================
@app.route('/stats/predictions/daily', methods=['GET'])
def get_daily_predictions():
    """Nombre de prédictions aujourd'hui"""
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(id) 
                FROM predictions 
                WHERE DATE(timestamp) = DATE('now')
            """)
            daily = cursor.fetchone()[0]

        return jsonify({
            "status": "success",
            "daily_predictions": daily,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Erreur stats journalières : {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5001, debug=False)

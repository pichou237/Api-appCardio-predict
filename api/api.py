# import pandas as pd
# import json
# import numpy as np
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# from datetime import datetime, timedelta
# import psycopg2
# import hashlib
# import os
# import re
# from werkzeug.security import generate_password_hash, check_password_hash
# import logging
# from flask_socketio import SocketIO, emit, disconnect, join_room
# import warnings
# from sklearn.exceptions import InconsistentVersionWarning
# from flask import request, jsonify
# from datetime import datetime
# import sendgrid
# from sendgrid import SendGridAPIClient
# from sendgrid.helpers.mail import Mail

# # from sendgrid.helpers.mail import Mail
# from itsdangerous import URLSafeTimedSerializer
# from flask import url_for
# import joblib  # important si pas encore importé



# # Suppress scikit-learn version warnings
# warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# # Configuration initiale
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})
# socketio = SocketIO(app, cors_allowed_origins="*")

# # Configuration
# app.config[
#     'DB_URI'] = 'postgresql://neondb_owner:npg_pHs9avSVlF7r@ep-steep-glade-aen8o7fn-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
# app.config['SECRET_KEY'] = os.urandom(24).hex()

# EMAIL_REGEX = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')

# # Chargement sécurisé du modèle
# # model = None
# # try:
# #     with open("finalized_model.sav", "rb") as f:
# #         model = pickle.load(f)
# #     logger.info("Modèle chargé avec succès")
# # except Exception as e:
# #     logger.error(f"Erreur de chargement du modèle : {str(e)}")
# #     raise



# # Chargement sécurisé du modèle cardio_model_cameroon
# model = None
# scaler = None
# label_encoders = None
# feature_names = None

# # try:
# #     model_data = joblib.load("./cardio_model_cameroon.pkl")
# #     model = model_data['model']
# #     scaler = model_data['scaler']
# #     label_encoders = model_data['label_encoders']
# #     feature_names = model_data['feature_names']
# #     logger.info("Modèle cardio_model_cameroon chargé avec succès")
# # except Exception as e:
# #     logger.error(f"Erreur de chargement du modèle : {str(e)}")
# #     raise

# # import os

# # # Obtenir le chemin absolu du fichier actuel (api.py)
# # current_dir = os.path.dirname(os.path.abspath(__file__))

# # # Construire le chemin absolu vers le modèle
# # model_path = os.path.join(current_dir, "cardio_model_cameroon.pkl")

# # # Charger le modèle
# # try:
# #     model_data = joblib.load(model_path)
# # except Exception as e:
# #     import logging
# #     logging.error(f"Erreur de chargement du modèle : {e}")
# #     raise

# import os
# import joblib
# import logging

# # Initialiser le logger si ce n'est pas déjà fait
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # Obtenir le chemin absolu du fichier actuel (api.py)
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Construire le chemin absolu vers le modèle
# model_path = os.path.join(current_dir, "cardio_model_cameroon.pkl")

# # Charger le modèle avec gestion d'erreurs
# try:
#     model_data = joblib.load(model_path)
#     model = model_data['model']
#     scaler = model_data['scaler']
#     label_encoders = model_data['label_encoders']
#     feature_names = model_data['feature_names']
#     logger.info("Modèle cardio_model_cameroon chargé avec succès")
# except Exception as e:
#     logger.error(f"Erreur de chargement du modèle : {str(e)}")
#     raise


# # Statistiques de normalisation
# mean_values = np.array([0.6821, 0.9635, 131.6, 246.5, 149.56, 0.327, 1.043, 1.397, 0.7185, 2.314])
# std_values = np.array([0.466, 1.032, 17.56, 51.75, 22.90, 0.470, 1.161, 0.616, 1.006, 0.613])

# # Configuration supplémentaire
# app.config['SECURITY_PASSWORD_SALT'] = os.urandom(24).hex()
# app.config['SENDGRID_API_KEY'] = 'your_sendgrid_api_key'
# app.config['SENDER_EMAIL'] = 'no-reply@heartdiseaseapp.com'
# app.config['FRONTEND_URL'] = 'https://votreapplication.com'  # URL de votre frontend

# # Initialisation SendGrid
# sg = SendGridAPIClient(api_key=app.config['SENDGRID_API_KEY'])


# # Serializer pour les tokens
# ts = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# # --- Fonctions Utilitaires ---
# def get_db():
#     try:
#         conn = psycopg2.connect(app.config['DB_URI'])
#         return conn
#     except Exception as e:
#         logger.error(f"Database connection error: {str(e)}")
#         raise


# def init_db():
#     """Initialise la structure de la base de données"""
#     commands = (
#         """
#         CREATE TABLE IF NOT EXISTS users (
#             id SERIAL PRIMARY KEY,
#             username VARCHAR(255) UNIQUE NOT NULL,
#             password_hash TEXT NOT NULL,
#             email VARCHAR(255) UNIQUE NOT NULL,
#             role VARCHAR(50) DEFAULT 'patient',
#             api_key VARCHAR(255) UNIQUE,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#         """,
#         """
#         CREATE TABLE IF NOT EXISTS predictions (
#             id SERIAL PRIMARY KEY,
#             user_id INTEGER REFERENCES users(id),
#             input_data JSONB,
#             prediction DECIMAL(5,2),
#             risk BOOLEAN,
#             timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#         """,
#         """
#         CREATE TABLE IF NOT EXISTS doctors (
#             id SERIAL PRIMARY KEY,
#             user_id INTEGER REFERENCES users(id),
#             license_number VARCHAR(100) UNIQUE,
#             specialties JSONB,
#             availability JSONB,
#             consultation_fee DECIMAL(10,2)
#         )
#         """,
#         """
#         CREATE TABLE IF NOT EXISTS appointments (
#             id SERIAL PRIMARY KEY,
#             patient_id INTEGER REFERENCES users(id),
#             doctor_id INTEGER REFERENCES doctors(id),
#             appointment_date TIMESTAMP NOT NULL,
#             duration INTEGER DEFAULT 30,
#             status VARCHAR(20) DEFAULT 'scheduled',
#             notes TEXT,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#         """,
#         """
#         CREATE TABLE IF NOT EXISTS notifications (
#             id SERIAL PRIMARY KEY,
#             user_id INTEGER REFERENCES users(id),
#             title VARCHAR(255) NOT NULL,
#             message TEXT NOT NULL,
#             is_read BOOLEAN DEFAULT FALSE,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#         """
#     )

#     try:
#         conn = get_db()
#         cur = conn.cursor()
#         for command in commands:
#             cur.execute(command)
#         conn.commit()
#         logger.info("Database initialized successfully")
#     except Exception as e:
#         logger.error(f"DB initialization error: {str(e)}")
#         if conn:
#             conn.rollback()
#         raise
#     finally:
#         if 'cur' in locals():
#             cur.close()
#         if 'conn' in locals():
#             conn.close()


# def generate_api_key(username):
#     return hashlib.sha256(f"{username}{datetime.now()}{app.config['SECRET_KEY']}".encode()).hexdigest()


# def validate_api_key(api_key):
#     """Vérifie si une clé API est valide"""
#     conn = None
#     cur = None
#     try:
#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute("SELECT id FROM users WHERE api_key = %s", (api_key,))
#         return cur.fetchone() is not None
#     except Exception as e:
#         logger.error(f"Erreur validation clé API: {str(e)}")
#         return False
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# # def get_prediction(data):
# #     if model is None:
# #         raise ValueError("Model not loaded")
# #
# #     data_array = np.array([float(x) for x in data])
# #     normalized_data = (data_array - mean_values) / std_values
# #     return round(model.predict_proba(normalized_data.reshape(1, -1))[0][0] * 100, 2)

# # def get_prediction(data_dict):
# #     if model is None:
# #         raise ValueError("Model not loaded")
# #
# #     # Vérifie que toutes les features attendues sont présentes
# #     missing_features = [f for f in feature_names if f not in data_dict]
# #     if missing_features:
# #         raise ValueError(f"Données manquantes : {', '.join(missing_features)}")
# #
# #     try:
# #         # Préparer l'array dans le bon ordre
# #         data_array = np.array([data_dict[feat] for feat in feature_names]).reshape(1, -1)
# #         # Normaliser
# #         data_scaled = scaler.transform(data_array)
# #         # Prédire
# #         prediction = model.predict(data_scaled)[0]
# #         proba = model.predict_proba(data_scaled)[0]
# #
# #         risk_level = label_encoders['risque_cardio'].inverse_transform([prediction])[0]
# #
# #         return {
# #             'risk_level': risk_level,
# #             'probabilities': proba.tolist()
# #         }
# #     except Exception as e:
# #         logger.error(f"Erreur de prédiction : {str(e)}")
# #         raise
# #

# # Fonction pour compléter les données manquantes
# def compute_derived_features(data_dict):
#     poids = float(data_dict.get("poids", 0))
#     taille_cm = float(data_dict.get("taille", 0))
#     taille_m = taille_cm / 100.0 if taille_cm else 0

#     imc = poids / (taille_m ** 2) if taille_m else 0
#     data_dict["imc"] = round(imc, 2)
#     data_dict["age_risk"] = 1 if int(data_dict.get("age", 0)) > 50 else 0
#     data_dict["imc_risk"] = 1 if imc > 30 else 0

#     risk_factors = [
#         'tabac', 'diabete_connu', 'stress', 'sedentarite',
#         'sommeil_moins_6h', 'sommeil_mauvaise_qualite',
#         'alimentation_grasse', 'activite_physique', 'antecedents_familiaux'
#     ]

#     count = 0
#     for key in risk_factors:
#         value = data_dict.get(key, "Non")
#         if value == "Oui" or (key == "activite_physique" and value == "Non"):
#             count += 1
#     data_dict["risk_factor_count"] = count

#     return data_dict


# def preprocess_input(data_dict):
#     processed = {}

#     for key, value in data_dict.items():
#         if key in label_encoders:
#             le = label_encoders[key]
#             try:
#                 processed[key] = le.transform([value])[0]
#             except ValueError:
#                 logger.warning(f"Valeur inconnue pour {key}: {value} — remplacée par 0")
#                 processed[key] = 0  # Valeur par défaut si valeur non vue pendant l'entraînement
#         else:
#             try:
#                 processed[key] = float(value)
#             except ValueError:
#                 logger.warning(f"Valeur non numérique inattendue pour {key}: {value} — remplacée par 0")
#                 processed[key] = 0

#     # Ajout des variables dérivées si ton modèle les attend
#     try:
#         taille_m = float(data_dict.get("taille", 0)) / 100
#         poids_kg = float(data_dict.get("poids", 0))
#         imc = round(poids_kg / (taille_m ** 2), 2) if taille_m > 0 else 0
#         processed["imc"] = imc
#         processed["age_risk"] = 1 if int(data_dict.get("age", 0)) >= 50 else 0
#         processed["imc_risk"] = 1 if imc > 25 else 0

#         # Compter le nombre de facteurs de risque "Oui"
#         risk_keys = [
#             "tabac", "alcool", "sedentarite", "stress", "sommeil_moins_6h",
#             "sommeil_mauvaise_qualite", "alimentation_grasse", "symptomes_diabete",
#             "maux_tete", "essoufflement", "douleurs_poitrine"
#         ]
#         risk_factor_count = sum(1 for key in risk_keys if data_dict.get(key) == "Oui")
#         processed["risk_factor_count"] = risk_factor_count
#     except Exception as e:
#         logger.warning(f"Erreur lors du calcul des variables dérivées : {str(e)}")

#     return processed



# # Fonction de prédiction
# def get_prediction(data_dict):
#     try:
#         if model is None:
#             raise ValueError("Model not loaded")

#         # Prétraitement des données
#         data_processed = preprocess_input(data_dict)

#         # Vérification des champs requis
#         missing_features = [f for f in feature_names if f not in data_processed]
#         if missing_features:
#             raise ValueError(f"Données manquantes : {', '.join(missing_features)}")

#         # Conversion en tableau numpy dans l'ordre des features attendues
#         data_array = np.array([data_processed[feat] for feat in feature_names]).reshape(1, -1)

#         # Normalisation
#         data_scaled = scaler.transform(data_array)

#         # Prédiction
#         prediction = model.predict(data_scaled)[0]
#         proba = model.predict_proba(data_scaled)[0]

#         # Interprétation du niveau de risque
#         risk_level = label_encoders['risque_cardio'].inverse_transform([prediction])[0]

#         return {
#             'risk_level': risk_level,
#             'probabilities': proba.tolist()
#         }
#     except Exception as e:
#         logger.error(f"Erreur de prédiction : {str(e)}")
#         raise


# def create_notification(user_id, title, message, notif_type=None, metadata=None):
#     conn = None
#     cur = None
#     try:
#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute('''
#             INSERT INTO notifications 
#                 (user_id, title, message, notification_type, metadata)
#             VALUES (%s, %s, %s, %s, %s)
#             RETURNING id, created_at
#         ''', (user_id, title, message, notif_type, json.dumps(metadata) if metadata else None))

#         notif_id, created_at = cur.fetchone()
#         conn.commit()

#         # Envoi en temps réel via SocketIO
#         socketio.emit('new_notification', {
#             'id': notif_id,
#             'title': title,
#             'message': message,
#             'type': notif_type,
#             'created_at': created_at.isoformat()
#         }, room=f'user_{user_id}')

#         return notif_id
#     except Exception as e:
#         logger.error(f"Erreur création notification: {str(e)}")
#         return None
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# def get_user_id_from_api_key(api_key):
#     """Récupère l'ID utilisateur à partir d'une clé API"""
#     conn = None
#     cur = None
#     try:
#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute('SELECT id FROM users WHERE api_key = %s', (api_key,))
#         result = cur.fetchone()
#         return result[0] if result else None
#     except Exception as e:
#         logger.error(f"Error getting user ID: {str(e)}")
#         return None
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# # --- Routes API ---

# @app.route('/register', methods=['POST'])
# def register():
#     """Enregistrement d'un nouvel utilisateur"""
#     try:
#         data = request.get_json()
#         required_fields = ['username', 'password', 'email']
#         if not all(field in data for field in required_fields):
#             return jsonify({"status": "error", "message": "Tous les champs sont requis"}), 400

#         if not EMAIL_REGEX.match(data['email']):
#             return jsonify({"status": "error", "message": "Email invalide"}), 400

#         password_hash = generate_password_hash(data['password'])
#         api_key = generate_api_key(data['username'])

#         conn = get_db()
#         cur = conn.cursor()
#         try:
#             cur.execute(
#                 "INSERT INTO users (username, password_hash, email, api_key) VALUES (%s, %s, %s, %s) RETURNING id",
#                 (data['username'], password_hash, data['email'], api_key)
#             )
#             user_id = cur.fetchone()[0]
#             conn.commit()

#             return jsonify({
#                 "status": "success",
#                 "message": "Utilisateur enregistré",
#                 "api_key": api_key,
#                 "user_id": user_id
#             }), 201
#         except psycopg2.IntegrityError as e:
#             conn.rollback()
#             if 'username' in str(e):
#                 return jsonify({"status": "error", "message": "Nom d'utilisateur déjà pris"}), 409
#             elif 'email' in str(e):
#                 return jsonify({"status": "error", "message": "Email déjà utilisé"}), 409
#             return jsonify({"status": "error", "message": "Erreur d'enregistrement"}), 400
#     except Exception as e:
#         logger.error(f"Erreur d'enregistrement : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur serveur"}), 500
#     finally:
#         if 'cur' in locals():
#             cur.close()
#         if 'conn' in locals():
#             conn.close()


# @app.route('/login', methods=['POST'])
# def login():
#     """Authentification utilisateur"""
#     conn = None
#     cur = None
#     try:
#         data = request.get_json()
#         if not data or 'email' not in data or 'password' not in data:
#             return jsonify({"status": "error", "message": "Nom d'utilisateur et mot de passe requis"}), 400

#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT id, password_hash, api_key, role 
#             FROM users 
#             WHERE email = %s
#         """, (data['email'],))

#         result = cur.fetchone()
#         if not result:
#             return jsonify({"status": "error", "message": "Identifiants invalides"}), 401

#         user_id, stored_hash, api_key, role = result
#         if check_password_hash(stored_hash, data['password']):
#             return jsonify({
#                 "status": "success",
#                 "api_key": api_key,
#                 "user_id": user_id,
#                 "role": role,
#                 "message": "Authentification réussie"
#             }), 200
#         else:
#             return jsonify({"status": "error", "message": "Mot de passe incorrect"}), 401
#     except Exception as e:
#         logger.error(f"Erreur de connexion : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur d'authentification"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     """Effectue une prédiction de risque cardiaque"""
# #     conn = None
# #     cur = None
# #     try:
# #         data = request.get_json()
# #         if not data or 'api_key' not in data or 'data' not in data:
# #             return jsonify({"status": "error", "message": "Clé API et données requises"}), 400
# #
# #         if not validate_api_key(data['api_key']):
# #             return jsonify({"status": "error", "message": "Clé API invalide"}), 403
# #
# #         # prediction = get_prediction(data['data'])
# #         # risk = bool(prediction > 45)
# #         result = get_prediction(data['data'])
# #         prediction = result['probabilities']  # ou une proba spécifique si tu veux
# #         risk_level = result['risk_level']
# #         risk = (risk_level == 'Élevé')
# #
# #         conn = get_db()
# #         cur = conn.cursor()
# #         # cur.execute(
# #         #     "INSERT INTO predictions (user_id, input_data, prediction, risk) "
# #         #     "VALUES ((SELECT id FROM users WHERE api_key = %s), %s, %s, %s) "
# #         #     "RETURNING id, timestamp",
# #         #     (data['api_key'], json.dumps(data['data']), prediction, risk)
# #         # )
# #
# #         cur.execute(
# #             "INSERT INTO predictions (user_id, input_data, prediction, risk) "
# #             "VALUES ((SELECT id FROM users WHERE api_key = %s), %s, %s, %s) "
# #             "RETURNING id, timestamp",
# #             (data['api_key'], json.dumps(data['data']), result['probabilities'][1], risk)
# #         )
# #
# #         pred_id, timestamp = cur.fetchone()
# #         conn.commit()
# #
# #         # Notification
# #         user_id = get_user_id_from_api_key(data['api_key'])
# #         if user_id:
# #             create_notification(
# #                 user_id,
# #                 "Résultat de prédiction",
# #                 f"Votre risque cardiaque est de {prediction}%",
# #                 "prediction",
# #                 {"prediction": prediction, "risk": risk}
# #             )
# #
# #         return jsonify({
# #             "status": "success",
# #             "prediction": prediction,
# #             "risk": risk,
# #             "timestamp": timestamp.isoformat(),
# #             "prediction_id": pred_id
# #         })
# #     except ValueError as e:
# #         return jsonify({"status": "error", "message": str(e)}), 400
# #     except Exception as e:
# #         logger.error(f"Erreur de prédiction : {str(e)}")
# #         return jsonify({"status": "error", "message": "Erreur lors de la prédiction"}), 500
# #     finally:
# #         if cur:
# #             cur.close()
# #         if conn:
# #             conn.close()


# # ✅ ROUTE /predict
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     conn = None
# #     cur = None
# #     try:
# #         data = request.get_json()
# #         if not data or 'api_key' not in data or 'data' not in data:
# #             return jsonify({"status": "error", "message": "Clé API et données requises"}), 400
# #
# #         if not validate_api_key(data['api_key']):
# #             return jsonify({"status": "error", "message": "Clé API invalide"}), 403
# #
# #         result = get_prediction(data['data'])
# #         prediction_proba = result['probabilities'][1]  # probabilité risque élevé
# #         risk_level = result['risk_level']
# #         risk = (risk_level == 'Élevé')
# #
# #         conn = get_db()
# #         cur = conn.cursor()
# #         cur.execute(
# #             "INSERT INTO predictions (user_id, input_data, prediction, risk) "
# #             "VALUES ((SELECT id FROM users WHERE api_key = %s), %s, %s, %s) "
# #             "RETURNING id, timestamp",
# #             (data['api_key'], json.dumps(data['data']), prediction_proba, risk)
# #         )
# #         pred_id, timestamp = cur.fetchone()
# #         conn.commit()
# #
# #         # Convertir numpy types en types natifs Python
# #         pred_id = int(pred_id)
# #         risk = bool(risk)
# #
# #         return jsonify({
# #             "status": "success",
# #             "prediction": prediction_proba,
# #             "risk_level": risk_level,
# #             "risk": risk,
# #             "timestamp": timestamp.isoformat(),
# #             "prediction_id": pred_id
# #         })
# #     except ValueError as e:
# #         return jsonify({"status": "error", "message": str(e)}), 400
# #     except Exception as e:
# #         logger.error(f"Erreur de prédiction : {str(e)}")
# #         return jsonify({"status": "error", "message": "Erreur lors de la prédiction"}), 500
# #     finally:
# #         if cur:
# #             cur.close()
# #         if conn:
# #             conn.close()

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     """Effectue une prédiction de risque cardiaque"""
# #     conn = None
# #     cur = None
# #     try:
# #         data = request.get_json()
# #
# #         if not data or 'api_key' not in data or 'data' not in data:
# #             return jsonify({"status": "error", "message": "Clé API et données requises"}), 400
# #
# #         if not validate_api_key(data['api_key']):
# #             return jsonify({"status": "error", "message": "Clé API invalide"}), 403
# #
# #         # Appel de la fonction de prédiction
# #         result = get_prediction(data['data'])
# #         prediction_score = result['probabilities'][1]  # probabilité d'être malade
# #         risk_level = result['risk_level']
# #         risk = (risk_level == 'Élevé')
# #
# #         conn = get_db()
# #         cur = conn.cursor()
# #
# #         # Insertion dans la base
# #         cur.execute(
# #             """
# #             INSERT INTO predictions (user_id, input_data, prediction, risk)
# #             VALUES ((SELECT id FROM users WHERE api_key = %s), %s, %s, %s)
# #             RETURNING id, timestamp
# #             """,
# #             (data['api_key'], json.dumps(data['data']), round(prediction_score * 100, 2), risk)
# #         )
# #
# #         row = cur.fetchone()
# #         conn.commit()
# #
# #         pred_id = row[0] if row else None
# #         timestamp = row[1].isoformat() if row and row[1] else datetime.utcnow().isoformat()
# #
# #         return jsonify({
# #             "status": "success",
# #             "prediction": round(prediction_score * 100, 2),
# #             "risk_level": risk_level,
# #             "risk": risk,
# #             "timestamp": timestamp,
# #             "prediction_id": pred_id
# #         })
# #
# #     except ValueError as e:
# #         return jsonify({"status": "error", "message": str(e)}), 400
# #
# #     except Exception as e:
# #         logger.error(f"Erreur de prédiction : {str(e)}")
# #         return jsonify({"status": "error", "message": "Erreur lors de la prédiction"}), 500
# #
# #     finally:
# #         if cur:
# #             cur.close()
# #         if conn:
# #             conn.close()


# from datetime import datetime

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Effectue une prédiction de risque cardiaque"""
#     conn = None
#     cur = None
#     try:
#         data = request.get_json()

#         if not data or 'api_key' not in data or 'data' not in data:
#             return jsonify({"status": "error", "message": "Clé API et données requises"}), 400

#         if not validate_api_key(data['api_key']):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403

#         # Appel de la fonction de prédiction
#         result = get_prediction(data['data'])
#         prediction_score = result['probabilities'][1]  # probabilité d'être malade
#         risk_level = result['risk_level']
#         risk = (risk_level == 'Élevé')

#         timestamp = datetime.utcnow()

#         conn = get_db()
#         cur = conn.cursor()

#         # Insertion dans la base
#         cur.execute(
#             """
#             INSERT INTO predictions (user_id, input_data, prediction, risk, timestamp)
#             VALUES ((SELECT id FROM users WHERE api_key = %s), %s, %s, %s, %s)
#             RETURNING id
#             """,
#             (data['api_key'], json.dumps(data['data']), round(prediction_score * 100, 2), risk, timestamp)
#         )

#         pred_id = cur.fetchone()[0]
#         conn.commit()

#         return jsonify({
#             "status": "success",
#             "prediction": round(prediction_score * 100, 2),
#             "risk_level": risk_level,
#             "risk": risk,
#             "timestamp": timestamp.isoformat(),
#             "prediction_id": pred_id
#         })

#     except ValueError as e:
#         return jsonify({"status": "error", "message": str(e)}), 400

#     except Exception as e:
#         logger.error(f"Erreur de prédiction : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur lors de la prédiction"}), 500

#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()



# # @app.route('/history', methods=['GET'])
# # def get_history():
# #     """Récupère l'historique des prédictions d'un utilisateur"""
# #     conn = None
# #     cur = None
# #     try:
# #         api_key = request.args.get('api_key')
# #         if not api_key:
# #             return jsonify({"status": "error", "message": "Clé API requise"}), 400
# #
# #         if not validate_api_key(api_key):
# #             return jsonify({"status": "error", "message": "Clé API invalide"}), 403
# #
# #         conn = get_db()
# #         cur = conn.cursor()
# #         cur.execute("""
# #             SELECT id, input_data, prediction, risk, timestamp
# #             FROM predictions
# #             WHERE user_id = (SELECT id FROM users WHERE api_key = %s)
# #             ORDER BY timestamp DESC
# #             LIMIT 20
# #         """, (api_key,))
# #
# #         history = []
# #         for row in cur.fetchall():
# #             history.append({
# #                 "id": row[0],
# #                 "input_data": json.loads(row[1]),
# #                 "prediction": float(row[2]),
# #                 "risk": bool(row[3]),
# #                 "timestamp": row[4].isoformat()
# #             })
# #
# #         return jsonify({
# #             "status": "success",
# #             "count": len(history),
# #             "history": history
# #         })
# #     except Exception as e:
# #         logger.error(f"Erreur historique : {str(e)}")
# #         return jsonify({"status": "error", "message": "Erreur de récupération"}), 500
# #     finally:
# #         if cur:
# #             cur.close()
# #         if conn:
# #             conn.close()

# @app.route('/history', methods=['GET'])
# def get_prediction_history():
#     """Retourne l’historique des prédictions pour un utilisateur"""
#     api_key = request.args.get('api_key')

#     if not api_key:
#         return jsonify({"status": "error", "message": "Clé API requise"}), 400

#     if not validate_api_key(api_key):
#         return jsonify({"status": "error", "message": "Clé API invalide"}), 403

#     conn = None
#     cur = None
#     try:
#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT id, input_data, prediction, risk, timestamp
#             FROM predictions
#             WHERE user_id = (SELECT id FROM users WHERE api_key = %s)
#             ORDER BY timestamp DESC
#         """, (api_key,))
#         rows = cur.fetchall()

#         history = []
#         for row in rows:
#             prediction_id, input_data, prediction, risk, timestamp = row
#             history.append({
#                 "prediction_id": prediction_id,
#                 "input_data": input_data,
#                 "prediction": float(prediction),
#                 "risk": risk,
#                 "timestamp": timestamp.isoformat() if timestamp else None
#             })

#         return jsonify({
#             "status": "success",
#             "history": history
#         })

#     except Exception as e:
#         logger.error(f"Erreur historique : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur lors de la récupération de l'historique"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()

# @app.route('/profile', methods=['GET'])
# def get_profile():
#     """Récupère le profil utilisateur"""
#     conn = None
#     cur = None
#     try:
#         api_key = request.args.get('api_key')
#         if not api_key:
#             return jsonify({"status": "error", "message": "Clé API requise"}), 400

#         if not validate_api_key(api_key):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403

#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT id, username, email, role, created_at
#             FROM users
#             WHERE api_key = %s
#         """, (api_key,))

#         user = cur.fetchone()
#         if not user:
#             return jsonify({"status": "error", "message": "Utilisateur non trouvé"}), 404

#         # Récupérer les stats des prédictions
#         cur.execute("""
#             SELECT 
#                 COUNT(*) as total_predictions,
#                 AVG(prediction) as average_risk,
#                 MAX(timestamp) as last_prediction
#             FROM predictions
#             WHERE user_id = %s
#         """, (user[0],))
#         stats = cur.fetchone()

#         return jsonify({
#             "status": "success",
#             "profile": {
#                 "id": user[0],
#                 "username": user[1],
#                 "email": user[2],
#                 "role": user[3],
#                 "created_at": user[4].isoformat(),
#                 "stats": {
#                     "total_predictions": stats[0] if stats[0] else 0,
#                     "average_risk": round(float(stats[1]), 2) if stats[1] else 0,
#                     "last_prediction": stats[2].isoformat() if stats[2] else None
#                 }
#             }
#         })
#     except Exception as e:
#         logger.error(f"Erreur profil : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur de récupération"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# @app.route('/profile', methods=['PUT'])
# def update_profile():
#     """Met à jour le profil utilisateur"""
#     conn = None
#     cur = None
#     try:
#         data = request.get_json()
#         if not data or 'api_key' not in data:
#             return jsonify({"status": "error", "message": "Clé API requise"}), 400

#         if not validate_api_key(data['api_key']):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403

#         updates = []
#         params = []
#         new_api_key = None

#         if 'email' in data:
#             if not EMAIL_REGEX.match(data['email']):
#                 return jsonify({"status": "error", "message": "Email invalide"}), 400
#             updates.append("email = %s")
#             params.append(data['email'])

#         if 'new_password' in data and 'current_password' in data:
#             conn = get_db()
#             cur = conn.cursor()
#             cur.execute("SELECT password_hash FROM users WHERE api_key = %s", (data['api_key'],))
#             result = cur.fetchone()
#             if not result or not check_password_hash(result[0], data['current_password']):
#                 return jsonify({"status": "error", "message": "Mot de passe actuel incorrect"}), 401

#             updates.append("password_hash = %s")
#             params.append(generate_password_hash(data['new_password']))

#         if 'username' in data:
#             # Vérifier si le nouveau username est disponible
#             cur.execute("SELECT id FROM users WHERE username = %s AND api_key != %s",
#                         (data['username'], data['api_key']))
#             if cur.fetchone():
#                 return jsonify({"status": "error", "message": "Nom d'utilisateur déjà pris"}), 409

#             updates.append("username = %s")
#             params.append(data['username'])
#             new_api_key = generate_api_key(data['username'])
#             updates.append("api_key = %s")
#             params.append(new_api_key)

#         if not updates:
#             return jsonify({"status": "error", "message": "Aucune modification fournie"}), 400

#         # Construction de la requête
#         params.append(data['api_key'])
#         query = f"UPDATE users SET {', '.join(updates)} WHERE api_key = %s RETURNING username, email"

#         cur.execute(query, params)
#         updated_user = cur.fetchone()
#         conn.commit()

#         response = {
#             "status": "success",
#             "message": "Profil mis à jour",
#             "username": updated_user[0],
#             "email": updated_user[1]
#         }
#         if new_api_key:
#             response['new_api_key'] = new_api_key

#         return jsonify(response), 200
#     except Exception as e:
#         logger.error(f"Erreur mise à jour profil : {str(e)}")
#         if conn:
#             conn.rollback()
#         return jsonify({"status": "error", "message": "Erreur de mise à jour"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# @app.route('/doctors', methods=['GET'])
# def list_doctors():
#     """Liste tous les médecins disponibles"""
#     conn = None
#     cur = None
#     try:
#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT d.id, u.username, u.email, d.specialties, d.availability, d.consultation_fee
#             FROM doctors d
#             JOIN users u ON d.user_id = u.id
#         """)

#         doctors = []
#         for row in cur.fetchall():
#             doctors.append({
#                 "id": row[0],
#                 "name": row[1],
#                 "email": row[2],
#                 "specialties": json.loads(row[3]) if row[3] else [],
#                 "availability": json.loads(row[4]) if row[4] else {},
#                 "fee": float(row[5]) if row[5] else 0
#             })

#         return jsonify({
#             "status": "success",
#             "count": len(doctors),
#             "doctors": doctors
#         })
#     except Exception as e:
#         logger.error(f"Erreur liste médecins : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur de récupération"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# @app.route('/appointments', methods=['GET'])
# def get_appointments():
#     """Récupère les rendez-vous d'un utilisateur"""
#     conn = None
#     cur = None
#     try:
#         api_key = request.args.get('api_key')
#         if not api_key:
#             return jsonify({"status": "error", "message": "Clé API requise"}), 400

#         if not validate_api_key(api_key):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403

#         conn = get_db()
#         cur = conn.cursor()

#         # Récupérer l'ID et le rôle de l'utilisateur
#         cur.execute("SELECT id, role FROM users WHERE api_key = %s", (api_key,))
#         user = cur.fetchone()
#         if not user:
#             return jsonify({"status": "error", "message": "Utilisateur non trouvé"}), 404

#         user_id, role = user

#         if role == 'patient':
#             cur.execute("""
#                 SELECT a.id, a.appointment_date, a.duration, a.status, a.notes,
#                        u.username as doctor_name, d.specialties
#                 FROM appointments a
#                 JOIN doctors d ON a.doctor_id = d.id
#                 JOIN users u ON d.user_id = u.id
#                 WHERE a.patient_id = %s
#                 ORDER BY a.appointment_date DESC
#             """, (user_id,))
#         else:  # Médecin
#             cur.execute("""
#                 SELECT a.id, a.appointment_date, a.duration, a.status, a.notes,
#                        u.username as patient_name
#                 FROM appointments a
#                 JOIN users u ON a.patient_id = u.id
#                 WHERE a.doctor_id = (SELECT id FROM doctors WHERE user_id = %s)
#                 ORDER BY a.appointment_date DESC
#             """, (user_id,))

#         appointments = []
#         for row in cur.fetchall():
#             appointments.append({
#                 "id": row[0],
#                 "date": row[1].isoformat(),
#                 "duration": row[2],
#                 "status": row[3],
#                 "notes": row[4],
#                 "with_user": row[5],  # Nom du médecin ou patient selon le rôle
#                 "specialties": json.loads(row[6]) if len(row) > 6 and row[6] else None
#             })

#         return jsonify({
#             "status": "success",
#             "appointments": appointments,
#             "role": role
#         })
#     except Exception as e:
#         logger.error(f"Erreur rendez-vous : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur de récupération"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# @app.route('/appointments', methods=['POST'])
# def create_appointment():
#     """Crée un nouveau rendez-vous"""
#     conn = None
#     cur = None
#     try:
#         data = request.get_json()
#         required = ['api_key', 'doctor_id', 'appointment_date']
#         if not all(k in data for k in required):
#             return jsonify({"status": "error", "message": "Champs manquants"}), 400

#         if not validate_api_key(data['api_key']):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403

#         appointment_date = datetime.fromisoformat(data['appointment_date'])
#         duration = data.get('duration', 30)

#         conn = get_db()
#         cur = conn.cursor()

#         # Vérifier si le créneau est disponible
#         cur.execute("""
#             SELECT 1 FROM appointments 
#             WHERE doctor_id = %s 
#             AND appointment_date = %s
#             AND status != 'canceled'
#         """, (data['doctor_id'], appointment_date))

#         if cur.fetchone():
#             return jsonify({"status": "error", "message": "Créneau indisponible"}), 409

#         # Créer le rendez-vous
#         cur.execute("""
#             INSERT INTO appointments (
#                 patient_id, doctor_id, appointment_date, duration, notes
#             ) VALUES (
#                 (SELECT id FROM users WHERE api_key = %s),
#                 %s, %s, %s, %s
#             ) RETURNING id
#         """, (
#             data['api_key'],
#             data['doctor_id'],
#             appointment_date,
#             duration,
#             data.get('notes', '')
#         ))

#         appointment_id = cur.fetchone()[0]
#         conn.commit()

#         # Récupérer les infos pour la notification
#         cur.execute("""
#             SELECT u.username, d.user_id 
#             FROM doctors d
#             JOIN users u ON d.user_id = u.id
#             WHERE d.id = %s
#         """, (data['doctor_id'],))
#         doctor_info = cur.fetchone()

#         # Envoyer des notifications
#         patient_id = get_user_id_from_api_key(data['api_key'])
#         if patient_id:
#             create_notification(
#                 patient_id,
#                 "Rendez-vous confirmé",
#                 f"Rendez-vous avec Dr. {doctor_info[0]} le {appointment_date.strftime('%d/%m/%Y à %H:%M')}",
#                 "appointment",
#                 {"appointment_id": appointment_id}
#             )

#         if doctor_info[1]:  # ID utilisateur du médecin
#             create_notification(
#                 doctor_info[1],
#                 "Nouveau rendez-vous",
#                 f"Rendez-vous le {appointment_date.strftime('%d/%m/%Y à %H:%M')}",
#                 "appointment",
#                 {"appointment_id": appointment_id}
#             )

#         return jsonify({
#             "status": "success",
#             "appointment_id": appointment_id,
#             "message": "Rendez-vous créé"
#         }), 201
#     except ValueError:
#         return jsonify({"status": "error", "message": "Format de date invalide"}), 400
#     except Exception as e:
#         logger.error(f"Erreur création RDV: {str(e)}")
#         if conn:
#             conn.rollback()
#         return jsonify({"status": "error", "message": "Erreur serveur"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# @app.route('/appointments/<int:appointment_id>', methods=['PUT'])
# def update_appointment(appointment_id):
#     """Met à jour un rendez-vous"""
#     conn = None
#     cur = None
#     try:
#         data = request.get_json()
#         if not data or 'api_key' not in data:
#             return jsonify({"status": "error", "message": "Clé API requise"}), 400

#         if not validate_api_key(data['api_key']):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403

#         conn = get_db()
#         cur = conn.cursor()

#         # Vérifier que l'utilisateur a le droit de modifier ce RDV
#         cur.execute("""
#             SELECT a.patient_id, a.doctor_id, d.user_id as doctor_user_id
#             FROM appointments a
#             LEFT JOIN doctors d ON a.doctor_id = d.id
#             WHERE a.id = %s
#         """, (appointment_id,))
#         appointment = cur.fetchone()

#         if not appointment:
#             return jsonify({"status": "error", "message": "Rendez-vous non trouvé"}), 404

#         patient_id, doctor_id, doctor_user_id = appointment
#         user_id = get_user_id_from_api_key(data['api_key'])

#         if user_id not in [patient_id, doctor_user_id]:
#             return jsonify({"status": "error", "message": "Non autorisé"}), 403

#         # Construire la requête de mise à jour
#         updates = []
#         params = []

#         if 'status' in data:
#             updates.append("status = %s")
#             params.append(data['status'])

#         if 'notes' in data:
#             updates.append("notes = %s")
#             params.append(data['notes'])

#         if 'appointment_date' in data:
#             new_date = datetime.fromisoformat(data['appointment_date'])
#             # Vérifier que le nouveau créneau est disponible
#             cur.execute("""
#                 SELECT 1 FROM appointments
#                 WHERE doctor_id = %s
#                 AND appointment_date = %s
#                 AND id != %s
#                 AND status != 'canceled'
#             """, (doctor_id, new_date, appointment_id))

#             if cur.fetchone():
#                 return jsonify({"status": "error", "message": "Créneau indisponible"}), 409

#             updates.append("appointment_date = %s")
#             params.append(new_date)

#         if not updates:
#             return jsonify({"status": "error", "message": "Aucune modification fournie"}), 400

#         params.append(appointment_id)
#         query = f"UPDATE appointments SET {', '.join(updates)} WHERE id = %s RETURNING appointment_date, status"

#         cur.execute(query, params)
#         updated_appointment = cur.fetchone()
#         conn.commit()

#         # Envoyer des notifications
#         if patient_id != user_id:  # Le médecin a modifié le RDV
#             create_notification(
#                 patient_id,
#                 "Rendez-vous modifié",
#                 f"Votre rendez-vous a été modifié: {updated_appointment[1]}",
#                 "appointment",
#                 {"appointment_id": appointment_id}
#             )
#         elif doctor_user_id:  # Le patient a modifié le RDV
#             create_notification(
#                 doctor_user_id,
#                 "Rendez-vous modifié",
#                 f"Rendez-vous modifié: {updated_appointment[1]}",
#                 "appointment",
#                 {"appointment_id": appointment_id}
#             )

#         return jsonify({
#             "status": "success",
#             "message": "Rendez-vous mis à jour",
#             "new_date": updated_appointment[0].isoformat(),
#             "new_status": updated_appointment[1]
#         })
#     except ValueError:
#         return jsonify({"status": "error", "message": "Format de date invalide"}), 400
#     except Exception as e:
#         logger.error(f"Erreur mise à jour RDV: {str(e)}")
#         if conn:
#             conn.rollback()
#         return jsonify({"status": "error", "message": "Erreur serveur"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# @app.route('/notifications', methods=['GET'])
# def get_notifications():
#     """Récupère les notifications d'un utilisateur"""
#     conn = None
#     cur = None
#     try:
#         api_key = request.args.get('api_key')
#         if not api_key or not validate_api_key(api_key):
#             return jsonify({"status": "error", "message": "Non autorisé"}), 403

#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT id, title, message, is_read, created_at, metadata
#             FROM notifications
#             WHERE user_id = (SELECT id FROM users WHERE api_key = %s)
#             ORDER BY created_at DESC
#             LIMIT 50
#         """, (api_key,))

#         notifications = []
#         for row in cur.fetchall():
#             notifications.append({
#                 "id": row[0],
#                 "title": row[1],
#                 "message": row[2],
#                 "is_read": row[3],
#                 "created_at": row[4].isoformat(),
#                 "metadata": json.loads(row[5]) if row[5] else None
#             })

#         return jsonify({
#             "status": "success",
#             "notifications": notifications
#         })
#     except Exception as e:
#         logger.error(f"Erreur récup. notifications: {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur serveur"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# @app.route('/notifications/<int:notification_id>', methods=['PUT'])
# def mark_notification_read(notification_id):
#     """Marque une notification comme lue"""
#     conn = None
#     cur = None
#     try:
#         api_key = request.args.get('api_key')
#         if not api_key or not validate_api_key(api_key):
#             return jsonify({"status": "error", "message": "Non autorisé"}), 403

#         conn = get_db()
#         cur = conn.cursor()

#         # Vérifier que la notification appartient à l'utilisateur
#         cur.execute("""
#             UPDATE notifications
#             SET is_read = TRUE
#             WHERE id = %s AND user_id = (SELECT id FROM users WHERE api_key = %s)
#             RETURNING id
#         """, (notification_id, api_key))

#         if not cur.fetchone():
#             return jsonify({"status": "error", "message": "Notification non trouvée"}), 404

#         conn.commit()
#         return jsonify({"status": "success", "message": "Notification marquée comme lue"})
#     except Exception as e:
#         logger.error(f"Erreur marquage notification: {str(e)}")
#         if conn:
#             conn.rollback()
#         return jsonify({"status": "error", "message": "Erreur serveur"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# @app.route('/stats/predictions', methods=['GET'])
# def get_prediction_stats():
#     """Récupère des statistiques sur les prédictions"""
#     conn = None
#     cur = None
#     try:
#         conn = get_db()
#         cur = conn.cursor()

#         # Statistiques globales
#         cur.execute("""
#             SELECT 
#                 COUNT(*) as total,
#                 AVG(prediction) as average_risk,
#                 COUNT(CASE WHEN risk = TRUE THEN 1 END) as high_risk_count
#             FROM predictions
#         """)
#         global_stats = cur.fetchone()

#         # Statistiques par jour (7 derniers jours)
#         cur.execute("""
#             SELECT 
#                 DATE(timestamp) as day,
#                 COUNT(*) as count,
#                 AVG(prediction) as average_risk
#             FROM predictions
#             WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
#             GROUP BY day
#             ORDER BY day
#         """)
#         daily_stats = []
#         for row in cur.fetchall():
#             daily_stats.append({
#                 "date": row[0].isoformat(),
#                 "count": row[1],
#                 "average_risk": float(row[2]) if row[2] else 0
#             })

#         return jsonify({
#             "status": "success",
#             "stats": {
#                 "total_predictions": global_stats[0],
#                 "average_risk": round(float(global_stats[1]), 2) if global_stats[1] else 0,
#                 "high_risk_percentage": round((global_stats[2] / global_stats[0] * 100), 2) if global_stats[
#                                                                                                    0] > 0 else 0,
#                 "daily_stats": daily_stats
#             }
#         })
#     except Exception as e:
#         logger.error(f"Erreur stats prédictions: {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur serveur"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# # --- WebSocket ---
# @socketio.on('connect')
# def handle_connect():
#     api_key = request.args.get('api_key')
#     if api_key and validate_api_key(api_key):
#         user_id = get_user_id_from_api_key(api_key)
#         if user_id:
#             emit('connection', {'status': 'connected', 'user_id': user_id})
#         else:
#             emit('connection', {'status': 'unauthorized'})
#             disconnect()
#     else:
#         emit('connection', {'status': 'unauthorized'})
#         disconnect()


# @socketio.on('join_notifications')
# def handle_join_notifications(data):
#     api_key = data.get('api_key')
#     if api_key and validate_api_key(api_key):
#         user_id = get_user_id_from_api_key(api_key)
#         if user_id:
#             join_room(f'user_{user_id}')
#             emit('notification_status', {'status': 'joined'})


# @app.route('/stats/users', methods=['GET'])
# def get_user_stats():
#     """Statistiques sur les utilisateurs"""
#     conn = None
#     cur = None
#     try:
#         conn = get_db()
#         cur = conn.cursor()

#         # Statistiques globales
#         cur.execute("""
#             SELECT 
#                 COUNT(*) as total_users,
#                 COUNT(CASE WHEN role = 'doctor' THEN 1 END) as doctors,
#                 COUNT(CASE WHEN role = 'patient' THEN 1 END) as patients,
#                 DATE(MIN(created_at)) as first_signup,
#                 COUNT(DISTINCT DATE(created_at)) as active_days
#             FROM users
#         """)
#         stats = cur.fetchone()

#         # Inscription par mois
#         cur.execute("""
#             SELECT 
#                 DATE_TRUNC('month', created_at) as month,
#                 COUNT(*) as new_users
#             FROM users
#             GROUP BY month
#             ORDER BY month
#         """)
#         monthly_stats = []
#         for row in cur.fetchall():
#             monthly_stats.append({
#                 "month": row[0].strftime("%Y-%m"),
#                 "new_users": row[1]
#             })

#         return jsonify({
#             "status": "success",
#             "stats": {
#                 "total_users": stats[0],
#                 "doctors": stats[1],
#                 "patients": stats[2],
#                 "first_signup": stats[3].isoformat() if stats[3] else None,
#                 "active_days": stats[4],
#                 "monthly_growth": monthly_stats
#             }
#         })
#     except Exception as e:
#         logger.error(f"Erreur stats utilisateurs: {str(e)}")
#         return jsonify({"status": "error", "message": str(e)}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# @app.route('/stats/activity', methods=['GET'])
# def get_activity_stats():
#     """Statistiques d'activité"""
#     conn = None
#     cur = None
#     try:
#         conn = get_db()
#         cur = conn.cursor()

#         # Activité récente
#         cur.execute("""
#             SELECT 
#                 COUNT(*) as total_predictions,
#                 COUNT(DISTINCT user_id) as active_users,
#                 MAX(timestamp) as last_activity
#             FROM predictions
#         """)
#         activity = cur.fetchone()

#         # Prédictions par heure de la journée
#         cur.execute("""
#             SELECT 
#                 EXTRACT(HOUR FROM timestamp) as hour,
#                 COUNT(*) as prediction_count
#             FROM predictions
#             GROUP BY hour
#             ORDER BY hour
#         """)
#         hourly_activity = []
#         for row in cur.fetchall():
#             hourly_activity.append({
#                 "hour": int(row[0]),
#                 "count": row[1]
#             })

#         return jsonify({
#             "status": "success",
#             "activity": {
#                 "total_predictions": activity[0],
#                 "active_users": activity[1],
#                 "last_activity": activity[2].isoformat() if activity[2] else None,
#                 "hourly_activity": hourly_activity
#             }
#         })
#     except Exception as e:
#         logger.error(f"Erreur stats activité: {str(e)}")
#         return jsonify({"status": "error", "message": str(e)}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# # ===== Email Provider (Connexion & Reset Password) =====

# def send_email(to_email, subject, html_content):
#     """Fonction utilitaire pour envoyer des emails"""
#     # message = Mail(
#     #     from_email=app.config['SENDER_EMAIL'],
#     #     to_emails=to_email,
#     #     subject=subject,
#     #     html_content=html_content)

#     message="hello worl"

#     try:
#         response = sg.send(message)
#         logger.info(f"Email envoyé à {to_email}, status: {response.status_code}")
#         return True
#     except Exception as e:
#         logger.error(f"Erreur envoi email: {str(e)}")
#         return False


# @app.route('/request-password-reset', methods=['POST'])
# def request_password_reset():
#     """Demande de réinitialisation de mot de passe"""
#     conn = None
#     cur = None
#     try:
#         email = request.json.get('email')
#         if not email:
#             return jsonify({"status": "error", "message": "Email requis"}), 400

#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute("SELECT id, username FROM users WHERE email = %s", (email,))
#         user = cur.fetchone()

#         if user:
#             user_id, username = user
#             # Générer un token valide 24h
#             token = ts.dumps(user_id, salt=app.config['SECURITY_PASSWORD_SALT'])

#             reset_url = f"{app.config['FRONTEND_URL']}/reset-password?token={token}"

#             # Envoyer l'email
#             html_content = f"""
#                 <h2>Réinitialisation de mot de passe</h2>
#                 <p>Bonjour {username},</p>
#                 <p>Vous avez demandé à réinitialiser votre mot de passe. Cliquez sur le lien ci-dessous :</p>
#                 <p><a href="{reset_url}">Réinitialiser mon mot de passe</a></p>
#                 <p>Ce lien expirera dans 24 heures.</p>
#                 <p>Si vous n'avez pas fait cette demande, ignorez simplement cet email.</p>
#             """

#             if send_email(email, "Réinitialisation de mot de passe", html_content):
#                 return jsonify({"status": "success", "message": "Email envoyé"})
#             else:
#                 return jsonify({"status": "error", "message": "Erreur d'envoi d'email"}), 500

#         # Pour des raisons de sécurité, on ne révèle pas si l'email existe
#         return jsonify({"status": "success", "message": "Si l'email existe, un lien de réinitialisation a été envoyé"})
#     except Exception as e:
#         logger.error(f"Erreur demande reset password: {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur serveur"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# @app.route('/reset-password', methods=['POST'])
# def reset_password():
#     """Réinitialisation du mot de passe avec token"""
#     conn = None
#     cur = None
#     try:
#         token = request.json.get('token')
#         new_password = request.json.get('new_password')

#         if not token or not new_password:
#             return jsonify({"status": "error", "message": "Token et nouveau mot de passe requis"}), 400

#         # Vérifier le token
#         try:
#             user_id = ts.loads(
#                 token,
#                 salt=app.config['SECURITY_PASSWORD_SALT'],
#                 max_age=86400  # 24 heures
#             )
#         except:
#             return jsonify({"status": "error", "message": "Token invalide ou expiré"}), 400

#         # Mettre à jour le mot de passe
#         password_hash = generate_password_hash(new_password)

#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute(
#             "UPDATE users SET password_hash = %s WHERE id = %s RETURNING email, username",
#             (password_hash, user_id)
#         )
#         user = cur.fetchone()

#         if not user:
#             return jsonify({"status": "error", "message": "Utilisateur non trouvé"}), 404

#         conn.commit()

#         # Envoyer une confirmation par email
#         email, username = user
#         html_content = f"""
#             <h2>Mot de passe mis à jour</h2>
#             <p>Bonjour {username},</p>
#             <p>Votre mot de passe a été modifié avec succès.</p>
#             <p>Si vous n'avez pas effectué cette modification, veuillez nous contacter immédiatement.</p>
#         """
#         send_email(email, "Confirmation de changement de mot de passe", html_content)

#         return jsonify({"status": "success", "message": "Mot de passe mis à jour"})
#     except Exception as e:
#         logger.error(f"Erreur reset password: {str(e)}")
#         if conn:
#             conn.rollback()
#         return jsonify({"status": "error", "message": "Erreur serveur"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()

# # --- Point d'entrée ---
# if __name__ == '__main__':
#     try:
#         init_db()
#         socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
#     except Exception as e:
#         logger.error(f"Application startup failed: {str(e)}")
#         raise


################################################################################################
############################### api.py v3 ######################################################
#################################################################################################
import pandas as pd
import json
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import psycopg2
import hashlib
import os
import re
from werkzeug.security import generate_password_hash, check_password_hash
import logging
from flask_socketio import SocketIO, emit, disconnect, join_room
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from flask import request, jsonify
from datetime import datetime
import sendgrid
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# from sendgrid.helpers.mail import Mail
from itsdangerous import URLSafeTimedSerializer
from flask import url_for
import joblib  # important si pas encore importé



# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Configuration initiale
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})
# socketio = SocketIO(app, cors_allowed_origins="*")

# # Autoriser uniquement ton frontend Vercel
# CORS(app, resources={r"/*": {"origins": "https://app-cardio.vercel.app" ,"http://localhost:8080"}})

# Autoriser uniquement ton frontend Vercel et le localhost
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://app-cardio.vercel.app",
            "http://localhost:8080"
        ]
    }
})

# Pour WebSockets
socketio = SocketIO(app, cors_allowed_origins="https://app-cardio.vercel.app")

# Configuration
app.config[
    'DB_URI'] = 'postgresql://neondb_owner:npg_pHs9avSVlF7r@ep-steep-glade-aen8o7fn-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
app.config['SECRET_KEY'] = os.urandom(24).hex()

EMAIL_REGEX = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')

# Chargement sécurisé du modèle
# model = None
# try:
#     with open("finalized_model.sav", "rb") as f:
#         model = pickle.load(f)
#     logger.info("Modèle chargé avec succès")
# except Exception as e:
#     logger.error(f"Erreur de chargement du modèle : {str(e)}")
#     raise



# Chargement sécurisé du modèle cardio_model_cameroon
model = None
scaler = None
label_encoders = None
feature_names = None

# try:
#     model_data = joblib.load("./cardio_model_cameroon.pkl")
#     model = model_data['model']
#     scaler = model_data['scaler']
#     label_encoders = model_data['label_encoders']
#     feature_names = model_data['feature_names']
#     logger.info("Modèle cardio_model_cameroon chargé avec succès")
# except Exception as e:
#     logger.error(f"Erreur de chargement du modèle : {str(e)}")
#     raise

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# # Construire le chemin absolu vers le modèle
model_path = os.path.join(current_dir, "cardio_model_cameroon.pkl")

# Charger le modèle avec gestion d'erreurs
try:
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']
    logger.info("Modèle cardio_model_cameroon chargé avec succès")
except Exception as e:
    logger.error(f"Erreur de chargement du modèle : {str(e)}")
    raise


# Statistiques de normalisation
mean_values = np.array([0.6821, 0.9635, 131.6, 246.5, 149.56, 0.327, 1.043, 1.397, 0.7185, 2.314])
std_values = np.array([0.466, 1.032, 17.56, 51.75, 22.90, 0.470, 1.161, 0.616, 1.006, 0.613])

# Configuration supplémentaire
app.config['SECURITY_PASSWORD_SALT'] = os.urandom(24).hex()
app.config['SENDGRID_API_KEY'] = 'your_sendgrid_api_key'
app.config['SENDER_EMAIL'] = 'no-reply@heartdiseaseapp.com'
app.config['FRONTEND_URL'] = 'https://votreapplication.com'  # URL de votre frontend

# Initialisation SendGrid
sg = SendGridAPIClient(api_key=app.config['SENDGRID_API_KEY'])


# Serializer pour les tokens
ts = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# --- Fonctions Utilitaires ---
def get_db():
    try:
        conn = psycopg2.connect(app.config['DB_URI'])
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise


def init_db():
    """Initialise la structure de la base de données"""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            role VARCHAR(50) DEFAULT 'patient',
            api_key VARCHAR(255) UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            input_data JSONB,
            prediction DECIMAL(5,2),
            risk BOOLEAN,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS doctors (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            license_number VARCHAR(100) UNIQUE,
            specialties JSONB,
            availability JSONB,
            consultation_fee DECIMAL(10,2)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS appointments (
            id SERIAL PRIMARY KEY,
            patient_id INTEGER REFERENCES users(id),
            doctor_id INTEGER REFERENCES doctors(id),
            appointment_date TIMESTAMP NOT NULL,
            duration INTEGER DEFAULT 30,
            status VARCHAR(20) DEFAULT 'scheduled',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            title VARCHAR(255) NOT NULL,
            message TEXT NOT NULL,
            is_read BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    try:
        conn = get_db()
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"DB initialization error: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


def generate_api_key(username):
    return hashlib.sha256(f"{username}{datetime.now()}{app.config['SECRET_KEY']}".encode()).hexdigest()


def validate_api_key(api_key):
    """Vérifie si une clé API est valide"""
    conn = None
    cur = None
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE api_key = %s", (api_key,))
        return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"Erreur validation clé API: {str(e)}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# def get_prediction(data):
#     if model is None:
#         raise ValueError("Model not loaded")
#
#     data_array = np.array([float(x) for x in data])
#     normalized_data = (data_array - mean_values) / std_values
#     return round(model.predict_proba(normalized_data.reshape(1, -1))[0][0] * 100, 2)

# def get_prediction(data_dict):
#     if model is None:
#         raise ValueError("Model not loaded")
#
#     # Vérifie que toutes les features attendues sont présentes
#     missing_features = [f for f in feature_names if f not in data_dict]
#     if missing_features:
#         raise ValueError(f"Données manquantes : {', '.join(missing_features)}")
#
#     try:
#         # Préparer l'array dans le bon ordre
#         data_array = np.array([data_dict[feat] for feat in feature_names]).reshape(1, -1)
#         # Normaliser
#         data_scaled = scaler.transform(data_array)
#         # Prédire
#         prediction = model.predict(data_scaled)[0]
#         proba = model.predict_proba(data_scaled)[0]
#
#         risk_level = label_encoders['risque_cardio'].inverse_transform([prediction])[0]
#
#         return {
#             'risk_level': risk_level,
#             'probabilities': proba.tolist()
#         }
#     except Exception as e:
#         logger.error(f"Erreur de prédiction : {str(e)}")
#         raise
#

# Fonction pour compléter les données manquantes
def compute_derived_features(data_dict):
    poids = float(data_dict.get("poids", 0))
    taille_cm = float(data_dict.get("taille", 0))
    taille_m = taille_cm / 100.0 if taille_cm else 0

    imc = poids / (taille_m ** 2) if taille_m else 0
    data_dict["imc"] = round(imc, 2)
    data_dict["age_risk"] = 1 if int(data_dict.get("age", 0)) > 50 else 0
    data_dict["imc_risk"] = 1 if imc > 30 else 0

    risk_factors = [
        'tabac', 'diabete_connu', 'stress', 'sedentarite',
        'sommeil_moins_6h', 'sommeil_mauvaise_qualite',
        'alimentation_grasse', 'activite_physique', 'antecedents_familiaux'
    ]

    count = 0
    for key in risk_factors:
        value = data_dict.get(key, "Non")
        if value == "Oui" or (key == "activite_physique" and value == "Non"):
            count += 1
    data_dict["risk_factor_count"] = count

    return data_dict


def preprocess_input(data_dict):
    processed = {}

    for key, value in data_dict.items():
        if key in label_encoders:
            le = label_encoders[key]
            try:
                processed[key] = le.transform([value])[0]
            except ValueError:
                logger.warning(f"Valeur inconnue pour {key}: {value} — remplacée par 0")
                processed[key] = 0  # Valeur par défaut si valeur non vue pendant l'entraînement
        else:
            try:
                processed[key] = float(value)
            except ValueError:
                logger.warning(f"Valeur non numérique inattendue pour {key}: {value} — remplacée par 0")
                processed[key] = 0

    # Ajout des variables dérivées si ton modèle les attend
    try:
        taille_m = float(data_dict.get("taille", 0)) / 100
        poids_kg = float(data_dict.get("poids", 0))
        imc = round(poids_kg / (taille_m ** 2), 2) if taille_m > 0 else 0
        processed["imc"] = imc
        processed["age_risk"] = 1 if int(data_dict.get("age", 0)) >= 50 else 0
        processed["imc_risk"] = 1 if imc > 25 else 0

        # Compter le nombre de facteurs de risque "Oui"
        risk_keys = [
            "tabac", "alcool", "sedentarite", "stress", "sommeil_moins_6h",
            "sommeil_mauvaise_qualite", "alimentation_grasse", "symptomes_diabete",
            "maux_tete", "essoufflement", "douleurs_poitrine"
        ]
        risk_factor_count = sum(1 for key in risk_keys if data_dict.get(key) == "Oui")
        processed["risk_factor_count"] = risk_factor_count
    except Exception as e:
        logger.warning(f"Erreur lors du calcul des variables dérivées : {str(e)}")

    return processed



# Fonction de prédiction
def get_prediction(data_dict):
    try:
        if model is None:
            raise ValueError("Model not loaded")

        # Prétraitement des données
        data_processed = preprocess_input(data_dict)

        # Vérification des champs requis
        missing_features = [f for f in feature_names if f not in data_processed]
        if missing_features:
            raise ValueError(f"Données manquantes : {', '.join(missing_features)}")

        # Conversion en tableau numpy dans l'ordre des features attendues
        data_array = np.array([data_processed[feat] for feat in feature_names]).reshape(1, -1)

        # Normalisation
        data_scaled = scaler.transform(data_array)

        # Prédiction
        prediction = model.predict(data_scaled)[0]
        proba = model.predict_proba(data_scaled)[0]

        # Interprétation du niveau de risque
        risk_level = label_encoders['risque_cardio'].inverse_transform([prediction])[0]

        return {
            'risk_level': risk_level,
            'probabilities': proba.tolist()
        }
    except Exception as e:
        logger.error(f"Erreur de prédiction : {str(e)}")
        raise


def preprocess_patient_data(patient_data):
    """Version unifiée du prétraitement avec toutes les fonctionnalités"""
    processed = patient_data.copy()

    # 1. Encodage des catégories (version robuste)
    categorical_features = [
        'sexe', 'ville', 'environnement', 'antecedents_familiaux',
        'tabac', 'alcool', 'activite_physique', 'sedentarite',
        'diabete_connu', 'symptomes_diabete', 'stress', 'sommeil_moins_6h',
        'sommeil_mauvaise_qualite', 'alimentation_grasse', 'fruits_legumes',
        'maux_tete', 'essoufflement', 'douleurs_poitrine'
    ]

    for feature in categorical_features:
        if feature in label_encoders:
            try:
                processed[feature] = label_encoders[feature].transform([str(processed.get(feature, ""))])[0]
            except ValueError:
                processed[feature] = 0
                logger.warning(f"Valeur inconnue pour {feature}, remplacée par 0")

    # 2. Calcul des variables dérivées (version complète)
    try:
        # Calcul IMC précis
        taille_m = float(processed.get("taille", 0)) / 100
        poids_kg = float(processed.get("poids", 0))
        processed["imc"] = round(poids_kg / (taille_m ** 2), 2) if taille_m > 0 else 0

        # Features de risque
        processed['age_risk'] = 1 if int(processed.get('age', 0)) > 50 else 0
        processed['imc_risk'] = 1 if processed['imc'] > 25 else 0

        # Comptage des facteurs de risque (version hybride)
        risk_factors = [
            'antecedents_familiaux', 'tabac', 'diabete_connu', 'stress',
            'sommeil_moins_6h', 'alimentation_grasse', 'maux_tete',
            'essoufflement', 'douleurs_poitrine'
        ]
        processed['risk_factor_count'] = sum(
            1 for factor in risk_factors
            if str(patient_data.get(factor, "")).lower() == "oui"
        )
    except Exception as e:
        logger.error(f"Erreur dans le calcul des variables dérivées : {e}")
        processed.update({
            'imc': 0,
            'age_risk': 0,
            'imc_risk': 0,
            'risk_factor_count': 0
        })

    # 3. Validation des features requises
    required_features = [f for f in feature_names if f not in ('age_risk', 'imc_risk', 'risk_factor_count')]
    missing = [f for f in required_features if f not in processed]

    if missing:
        raise ValueError(f"Features manquantes : {missing}")

    return processed


def get_prediction(patient_data):
    """Version finale unifiée"""
    try:
        if model is None or scaler is None:
            raise ValueError("Modèle ou scaler non chargé")

        # Prétraitement complet
        processed = preprocess_patient_data(patient_data)

        # Construction du vecteur dans le bon ordre
        feature_vector = [processed.get(f, 0) for f in feature_names]

        # Prédiction
        scaled_data = scaler.transform([feature_vector])
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]

        return {
            'risk_level': label_encoders['risque_cardio'].inverse_transform([prediction])[0],
            'confidence': round(max(probabilities) * 100, 2),
            'probabilities': probabilities.tolist(),
            'risk_factors_count': processed['risk_factor_count'],
            'derived_features': {
                'imc': processed.get('imc', 0),
                'age_risk': processed.get('age_risk', 0),
                'imc_risk': processed.get('imc_risk', 0)
            }
        }

    except ValueError as ve:
        logger.error(f"Erreur de validation : {ve}")
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
        raise RuntimeError("Erreur lors de la prédiction")



def create_notification(user_id, title, message, notif_type=None, metadata=None):
    conn = None
    cur = None
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO notifications 
                (user_id, title, message, notification_type, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, created_at
        ''', (user_id, title, message, notif_type, json.dumps(metadata) if metadata else None))

        notif_id, created_at = cur.fetchone()
        conn.commit()

        # Envoi en temps réel via SocketIO
        socketio.emit('new_notification', {
            'id': notif_id,
            'title': title,
            'message': message,
            'type': notif_type,
            'created_at': created_at.isoformat()
        }, room=f'user_{user_id}')

        return notif_id
    except Exception as e:
        logger.error(f"Erreur création notification: {str(e)}")
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def get_user_id_from_api_key(api_key):
    """Récupère l'ID utilisateur à partir d'une clé API"""
    conn = None
    cur = None
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute('SELECT id FROM users WHERE api_key = %s', (api_key,))
        result = cur.fetchone()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error getting user ID: {str(e)}")
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# --- Routes API ---

@app.route('/register', methods=['POST'])
def register():
    """Enregistrement d'un nouvel utilisateur"""
    try:
        data = request.get_json()
        required_fields = ['username', 'password', 'email']
        if not all(field in data for field in required_fields):
            return jsonify({"status": "error", "message": "Tous les champs sont requis"}), 400

        if not EMAIL_REGEX.match(data['email']):
            return jsonify({"status": "error", "message": "Email invalide"}), 400

        password_hash = generate_password_hash(data['password'])
        api_key = generate_api_key(data['username'])

        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, password_hash, email, api_key ,role) VALUES (%s, %s, %s, %s,%s) RETURNING id",
                (data['username'], password_hash, data['email'], api_key ,"Utilisateur")
            )
            user_id = cur.fetchone()[0]
            conn.commit()

            return jsonify({
                "status": "success",
                "message": "Utilisateur enregistré",
                "api_key": api_key,
                "user_id": user_id
            }), 201
        except psycopg2.IntegrityError as e:
            conn.rollback()
            if 'username' in str(e):
                return jsonify({"status": "error", "message": "Nom d'utilisateur déjà pris"}), 409
            elif 'email' in str(e):
                return jsonify({"status": "error", "message": "Email déjà utilisé"}), 409
            return jsonify({"status": "error", "message": "Erreur d'enregistrement"}), 400
    except Exception as e:
        logger.error(f"Erreur d'enregistrement : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur serveur"}), 500
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


@app.route('/login', methods=['POST'])
def login():
    """Authentification utilisateur"""
    conn = None
    cur = None
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"status": "error", "message": "Nom d'utilisateur et mot de passe requis"}), 400

        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, password_hash, api_key, role 
            FROM users 
            WHERE email = %s
        """, (data['email'],))

        result = cur.fetchone()
        if not result:
            return jsonify({"status": "error", "message": "Identifiants invalides"}), 401

        user_id, stored_hash, api_key, role = result
        if check_password_hash(stored_hash, data['password']):
            return jsonify({
                "status": "success",
                "api_key": api_key,
                "user_id": user_id,
                "role": role,
                "message": "Authentification réussie"
            }), 200
        else:
            return jsonify({"status": "error", "message": "Mot de passe incorrect"}), 401
    except Exception as e:
        logger.error(f"Erreur de connexion : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur d'authentification"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# @app.route('/predict', methods=['POST'])
# def predict():
#     """Effectue une prédiction de risque cardiaque"""
#     conn = None
#     cur = None
#     try:
#         data = request.get_json()
#         if not data or 'api_key' not in data or 'data' not in data:
#             return jsonify({"status": "error", "message": "Clé API et données requises"}), 400
#
#         if not validate_api_key(data['api_key']):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403
#
#         # prediction = get_prediction(data['data'])
#         # risk = bool(prediction > 45)
#         result = get_prediction(data['data'])
#         prediction = result['probabilities']  # ou une proba spécifique si tu veux
#         risk_level = result['risk_level']
#         risk = (risk_level == 'Élevé')
#
#         conn = get_db()
#         cur = conn.cursor()
#         # cur.execute(
#         #     "INSERT INTO predictions (user_id, input_data, prediction, risk) "
#         #     "VALUES ((SELECT id FROM users WHERE api_key = %s), %s, %s, %s) "
#         #     "RETURNING id, timestamp",
#         #     (data['api_key'], json.dumps(data['data']), prediction, risk)
#         # )
#
#         cur.execute(
#             "INSERT INTO predictions (user_id, input_data, prediction, risk) "
#             "VALUES ((SELECT id FROM users WHERE api_key = %s), %s, %s, %s) "
#             "RETURNING id, timestamp",
#             (data['api_key'], json.dumps(data['data']), result['probabilities'][1], risk)
#         )
#
#         pred_id, timestamp = cur.fetchone()
#         conn.commit()
#
#         # Notification
#         user_id = get_user_id_from_api_key(data['api_key'])
#         if user_id:
#             create_notification(
#                 user_id,
#                 "Résultat de prédiction",
#                 f"Votre risque cardiaque est de {prediction}%",
#                 "prediction",
#                 {"prediction": prediction, "risk": risk}
#             )
#
#         return jsonify({
#             "status": "success",
#             "prediction": prediction,
#             "risk": risk,
#             "timestamp": timestamp.isoformat(),
#             "prediction_id": pred_id
#         })
#     except ValueError as e:
#         return jsonify({"status": "error", "message": str(e)}), 400
#     except Exception as e:
#         logger.error(f"Erreur de prédiction : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur lors de la prédiction"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# ✅ ROUTE /predict
# @app.route('/predict', methods=['POST'])
# def predict():
#     conn = None
#     cur = None
#     try:
#         data = request.get_json()
#         if not data or 'api_key' not in data or 'data' not in data:
#             return jsonify({"status": "error", "message": "Clé API et données requises"}), 400
#
#         if not validate_api_key(data['api_key']):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403
#
#         result = get_prediction(data['data'])
#         prediction_proba = result['probabilities'][1]  # probabilité risque élevé
#         risk_level = result['risk_level']
#         risk = (risk_level == 'Élevé')
#
#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute(
#             "INSERT INTO predictions (user_id, input_data, prediction, risk) "
#             "VALUES ((SELECT id FROM users WHERE api_key = %s), %s, %s, %s) "
#             "RETURNING id, timestamp",
#             (data['api_key'], json.dumps(data['data']), prediction_proba, risk)
#         )
#         pred_id, timestamp = cur.fetchone()
#         conn.commit()
#
#         # Convertir numpy types en types natifs Python
#         pred_id = int(pred_id)
#         risk = bool(risk)
#
#         return jsonify({
#             "status": "success",
#             "prediction": prediction_proba,
#             "risk_level": risk_level,
#             "risk": risk,
#             "timestamp": timestamp.isoformat(),
#             "prediction_id": pred_id
#         })
#     except ValueError as e:
#         return jsonify({"status": "error", "message": str(e)}), 400
#     except Exception as e:
#         logger.error(f"Erreur de prédiction : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur lors de la prédiction"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Effectue une prédiction de risque cardiaque"""
#     conn = None
#     cur = None
#     try:
#         data = request.get_json()
#
#         if not data or 'api_key' not in data or 'data' not in data:
#             return jsonify({"status": "error", "message": "Clé API et données requises"}), 400
#
#         if not validate_api_key(data['api_key']):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403
#
#         # Appel de la fonction de prédiction
#         result = get_prediction(data['data'])
#         prediction_score = result['probabilities'][1]  # probabilité d'être malade
#         risk_level = result['risk_level']
#         risk = (risk_level == 'Élevé')
#
#         conn = get_db()
#         cur = conn.cursor()
#
#         # Insertion dans la base
#         cur.execute(
#             """
#             INSERT INTO predictions (user_id, input_data, prediction, risk)
#             VALUES ((SELECT id FROM users WHERE api_key = %s), %s, %s, %s)
#             RETURNING id, timestamp
#             """,
#             (data['api_key'], json.dumps(data['data']), round(prediction_score * 100, 2), risk)
#         )
#
#         row = cur.fetchone()
#         conn.commit()
#
#         pred_id = row[0] if row else None
#         timestamp = row[1].isoformat() if row and row[1] else datetime.utcnow().isoformat()
#
#         return jsonify({
#             "status": "success",
#             "prediction": round(prediction_score * 100, 2),
#             "risk_level": risk_level,
#             "risk": risk,
#             "timestamp": timestamp,
#             "prediction_id": pred_id
#         })
#
#     except ValueError as e:
#         return jsonify({"status": "error", "message": str(e)}), 400
#
#     except Exception as e:
#         logger.error(f"Erreur de prédiction : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur lors de la prédiction"}), 500
#
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


# Dans la route Flask :
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#
#         # Validation de base
#         if not data or 'api_key' not in data or 'data' not in data:
#             return jsonify({"status": "error", "message": "Clé API et données requises"}), 400
#
#         # Utilisez data['data'] au lieu de data['patient']
#         result = get_prediction(data['data'])  # <-- Correction ici
#
#         # Le reste de votre logique...
#         prediction_score = result['probabilities'][1]
#         risk_level = result['risk_level']
#
#         return jsonify({
#             "status": "success",
#             "prediction": round(prediction_score * 100, 2),
#             "risk_level": risk_level,
#             "risk": risk_level == "Élevé"
#         })
#
#     except KeyError as e:
#         logger.error(f"Champ manquant dans les données : {str(e)}")
#         return jsonify({"status": "error", "message": f"Champ requis manquant : {str(e)}"}), 400
#     except Exception as e:
#         logger.error(f"Erreur de prédiction : {str(e)}", exc_info=True)
#         return jsonify({"status": "error", "message": "Erreur lors de la prédiction"}), 500
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#
#         # Validation requise
#         if not data or 'api_key' not in data or 'data' not in data:
#             return jsonify({"status": "error", "message": "API key and data required"}), 400
#
#         # Prédiction
#         result = get_prediction(data['data'])
#
#         # Réponse standardisée
#         return jsonify({
#             "status": "success",
#             "prediction_details": {
#                 "risk_level": result['risk_level'],  # "Élevé"/"Moyen"/"Faible"
#                 "confidence": result['confidence'],  # Pourcentage (0-100)
#                 "probabilities": result['probabilities'],  # [Faible, Moyen, Élevé]
#                 "risk_factors_count": result['risk_factors_count'],
#                 "derived_features": result.get('derived_features', {})
#             },
#             "risk": result['risk_level'] == "Élevé"  # Booléen pour compatibilité
#         })
#
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}", exc_info=True)
#         return jsonify({"status": "error", "message": str(e)}), 500

from datetime import datetime

@app.route('/predict', methods=['POST'])
def predict():
    """Effectue une prédiction de risque cardiaque"""
    conn = None
    cur = None
    try:
        data = request.get_json()

        if not data or 'api_key' not in data or 'data' not in data:
            return jsonify({"status": "error", "message": "Clé API et données requises"}), 400

        if not validate_api_key(data['api_key']):
            return jsonify({"status": "error", "message": "Clé API invalide"}), 403

        # Appel de la fonction de prédiction
        result = get_prediction(data['data'])
        # prediction_score = result['probabilities'][1]  # probabilité d'être malade
        prediction_score = max(result['probabilities'])
        risk_level = result['risk_level']

        probabilities_json = json.dumps(result['probabilities'])  # Conversion des probabilités en JSON

        # Version correcte
        risk = risk_level in {'Élevé', 'Modéré'}  # Utilisation d'un set pour l'efficacité

        timestamp = datetime.utcnow()

        conn = get_db()
        cur = conn.cursor()

        # Insertion dans la base
        cur.execute(
            """
            INSERT INTO predictions (user_id, input_data, prediction, risk, timestamp,risk_level,probabilities)
            VALUES ((SELECT id FROM users WHERE api_key = %s), %s, %s, %s, %s,%s,%s)
            RETURNING id
            """,
            (data['api_key'], json.dumps(data['data']), round(prediction_score * 100, 2), risk, timestamp,risk_level,probabilities_json)
        )

        pred_id = cur.fetchone()[0]
        conn.commit()

        return jsonify({
            "status": "success",
            "prediction": round(prediction_score * 100, 3) , # 96.876 au lieu de (96.876, 3),
            "risk_level": risk_level,
            "risk": risk,
            "probabilities":probabilities_json,
            "timestamp": timestamp.isoformat(),
            "prediction_id": pred_id
        })

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    except Exception as e:
        logger.error(f"Erreur de prédiction : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur lors de la prédiction"}), 500

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()



# @app.route('/history', methods=['GET'])
# def get_history():
#     """Récupère l'historique des prédictions d'un utilisateur"""
#     conn = None
#     cur = None
#     try:
#         api_key = request.args.get('api_key')
#         if not api_key:
#             return jsonify({"status": "error", "message": "Clé API requise"}), 400
#
#         if not validate_api_key(api_key):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403
#
#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT id, input_data, prediction, risk, timestamp
#             FROM predictions
#             WHERE user_id = (SELECT id FROM users WHERE api_key = %s)
#             ORDER BY timestamp DESC
#             LIMIT 20
#         """, (api_key,))
#
#         history = []
#         for row in cur.fetchall():
#             history.append({
#                 "id": row[0],
#                 "input_data": json.loads(row[1]),
#                 "prediction": float(row[2]),
#                 "risk": bool(row[3]),
#                 "timestamp": row[4].isoformat()
#             })
#
#         return jsonify({
#             "status": "success",
#             "count": len(history),
#             "history": history
#         })
#     except Exception as e:
#         logger.error(f"Erreur historique : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur de récupération"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()

@app.route('/history', methods=['GET'])
def get_prediction_history():
    """Retourne l’historique des prédictions pour un utilisateur"""
    api_key = request.args.get('api_key')

    if not api_key:
        return jsonify({"status": "error", "message": "Clé API requise"}), 400

    if not validate_api_key(api_key):
        return jsonify({"status": "error", "message": "Clé API invalide"}), 403

    conn = None
    cur = None
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, input_data, prediction, risk, timestamp
            FROM predictions
            WHERE user_id = (SELECT id FROM users WHERE api_key = %s)
            ORDER BY timestamp DESC
        """, (api_key,))
        rows = cur.fetchall()

        history = []
        for row in rows:
            prediction_id, input_data, prediction, risk, timestamp = row
            history.append({
                "prediction_id": prediction_id,
                "input_data": input_data,
                "prediction": float(prediction),
                "risk": risk,
                "timestamp": timestamp.isoformat() if timestamp else None
            })

        return jsonify({
            "status": "success",
            "history": history
        })

    except Exception as e:
        logger.error(f"Erreur historique : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur lors de la récupération de l'historique"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# @app.route('/profile', methods=['GET'])
# def get_profile():
#     """Récupère le profil utilisateur"""
#     conn = None
#     cur = None
#     try:
#         api_key = request.args.get('api_key')
#         if not api_key:
#             return jsonify({"status": "error", "message": "Clé API requise"}), 400

#         if not validate_api_key(api_key):
#             return jsonify({"status": "error", "message": "Clé API invalide"}), 403

#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT id, username, email, role, created_at
#             FROM users
#             WHERE api_key = %s
#         """, (api_key,))

#         user = cur.fetchone()
#         if not user:
#             return jsonify({"status": "error", "message": "Utilisateur non trouvé"}), 404

#         # Récupérer les stats des prédictions
#         cur.execute("""
#             SELECT 
#                 COUNT(*) as total_predictions,
#                 AVG(prediction) as average_risk,
#                 MAX(timestamp) as last_prediction
#             FROM predictions
#             WHERE user_id = %s
#         """, (user[0],))
#         stats = cur.fetchone()

#         return jsonify({
#             "status": "success",
#             "profile": {
#                 "id": user[0],
#                 "username": user[1],
#                 "email": user[2],
#                 "role": user[3],
#                 "created_at": user[4].isoformat(),
#                 "stats": {
#                     "total_predictions": stats[0] if stats[0] else 0,
#                     "average_risk": round(float(stats[1]), 2) if stats[1] else 0,
#                     "last_prediction": stats[2].isoformat() if stats[2] else None
#                 }
#             }
#         })
#     except Exception as e:
#         logger.error(f"Erreur profil : {str(e)}")
#         return jsonify({"status": "error", "message": "Erreur de récupération"}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             conn.close()


@app.route('/profile', methods=['GET'])
def get_profile():
    """Récupère le profil utilisateur"""
    conn = None
    cur = None
    try:
        api_key = request.args.get('api_key')
        if not api_key:
            return jsonify({"status": "error", "message": "Clé API requise"}), 400

        if not validate_api_key(api_key):
            return jsonify({"status": "error", "message": "Clé API invalide"}), 403

        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, username, email, role, created_at
            FROM users
            WHERE api_key = %s
        """, (api_key,))

        user = cur.fetchone()
        if not user:
            return jsonify({"status": "error", "message": "Utilisateur non trouvé"}), 404

        user_id = user[0]
        username = user[1]
        email = user[2]
        role = user[3]
        created_at = user[4]

        # Récupérer les stats des prédictions
        cur.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                AVG(prediction) as average_risk,
                MAX(timestamp) as last_prediction
            FROM predictions
            WHERE user_id = %s
        """, (user_id,))
        stats = cur.fetchone()

        return jsonify({
            "status": "success",
            "profile": {
                "id": user_id,
                "username": username,
                "email": email,
                "role": role,
                "created_at": created_at.isoformat() if created_at else None,
                "stats": {
                    "total_predictions": stats[0] if stats[0] else 0,
                    "average_risk": round(float(stats[1]), 2) if stats[1] else 0,
                    "last_prediction": stats[2].isoformat() if stats[2] else None
                }
            }
        })
    except Exception as e:
        logger.error(f"Erreur profil : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur de récupération"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@app.route('/profile', methods=['PUT'])
def update_profile():
    """Met à jour le profil utilisateur"""
    conn = None
    cur = None
    try:
        data = request.get_json()
        if not data or 'api_key' not in data:
            return jsonify({"status": "error", "message": "Clé API requise"}), 400

        if not validate_api_key(data['api_key']):
            return jsonify({"status": "error", "message": "Clé API invalide"}), 403

        updates = []
        params = []
        new_api_key = None

        if 'email' in data:
            if not EMAIL_REGEX.match(data['email']):
                return jsonify({"status": "error", "message": "Email invalide"}), 400
            updates.append("email = %s")
            params.append(data['email'])

        if 'new_password' in data and 'current_password' in data:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT password_hash FROM users WHERE api_key = %s", (data['api_key'],))
            result = cur.fetchone()
            if not result or not check_password_hash(result[0], data['current_password']):
                return jsonify({"status": "error", "message": "Mot de passe actuel incorrect"}), 401

            updates.append("password_hash = %s")
            params.append(generate_password_hash(data['new_password']))

        if 'username' in data:
            # Vérifier si le nouveau username est disponible
            cur.execute("SELECT id FROM users WHERE username = %s AND api_key != %s",
                        (data['username'], data['api_key']))
            if cur.fetchone():
                return jsonify({"status": "error", "message": "Nom d'utilisateur déjà pris"}), 409

            updates.append("username = %s")
            params.append(data['username'])
            new_api_key = generate_api_key(data['username'])
            updates.append("api_key = %s")
            params.append(new_api_key)

        if not updates:
            return jsonify({"status": "error", "message": "Aucune modification fournie"}), 400

        # Construction de la requête
        params.append(data['api_key'])
        query = f"UPDATE users SET {', '.join(updates)} WHERE api_key = %s RETURNING username, email"

        cur.execute(query, params)
        updated_user = cur.fetchone()
        conn.commit()

        response = {
            "status": "success",
            "message": "Profil mis à jour",
            "username": updated_user[0],
            "email": updated_user[1]
        }
        if new_api_key:
            response['new_api_key'] = new_api_key

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Erreur mise à jour profil : {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({"status": "error", "message": "Erreur de mise à jour"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()



@app.route('/api/statistics/user', methods=['GET'])
def get_user_prediction_stats():
    conn = None
    cur = None
    try:
        api_key = request.args.get('api_key')
        if not api_key:
            return jsonify({"error": "Clé API requise"}), 400

        if not validate_api_key(api_key):
            return jsonify({"error": "Clé API invalide"}), 403

        conn = get_db()
        cur = conn.cursor()

        # Récupérer l'id de l'utilisateur
        cur.execute("SELECT id FROM users WHERE api_key = %s", (api_key,))
        user = cur.fetchone()
        if not user:
            return jsonify({"error": "Utilisateur non trouvé"}), 404
        user_id = user[0]

        # Total de prédictions
        cur.execute("SELECT COUNT(*) FROM predictions WHERE user_id = %s", (user_id,))
        total_predictions = cur.fetchone()[0] or 0

        # Prédictions du mois
        start_month = datetime.today().replace(day=1)
        cur.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE user_id = %s AND timestamp >= %s
        """, (user_id, start_month))
        monthly_predictions = cur.fetchone()[0] or 0

        # Prédictions aujourd’hui
        start_day = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        cur.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE user_id = %s AND timestamp >= %s
        """, (user_id, start_day))
        daily_predictions = cur.fetchone()[0] or 0

        # Moyenne du risque
        cur.execute("SELECT AVG(prediction) FROM predictions WHERE user_id = %s", (user_id,))
        average = cur.fetchone()[0]
        average_risk = round(float(average), 2) if average else 0

        # Dernière prédiction
        cur.execute("""
            SELECT prediction, timestamp 
            FROM predictions 
            WHERE user_id = %s 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (user_id,))
        last = cur.fetchone()
        last_prediction = round(float(last[0]), 2) if last else None
        last_prediction_time = last[1].strftime('%Y-%m-%d %H:%M:%S') if last else None

        return jsonify({
            "totalPredictions": total_predictions,
            "monthlyPredictions": monthly_predictions,
            "dailyPredictions": daily_predictions,
            "averageRisk": average_risk,
            "lastPrediction": last_prediction,
            "lastPredictionTime": last_prediction_time
        })
    except Exception as e:
        logger.error(f"Erreur stats utilisateur : {str(e)}")
        return jsonify({"error": "Erreur lors de la récupération"}), 500
    finally:
        if cur: cur.close()
        if conn: conn.close()



@app.route('/doctors', methods=['GET'])
def list_doctors():
    """Liste tous les médecins disponibles"""
    conn = None
    cur = None
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT d.id, u.username, u.email, d.specialties, d.availability, d.consultation_fee
            FROM doctors d
            JOIN users u ON d.user_id = u.id
        """)

        doctors = []
        for row in cur.fetchall():
            doctors.append({
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "specialties": json.loads(row[3]) if row[3] else [],
                "availability": json.loads(row[4]) if row[4] else {},
                "fee": float(row[5]) if row[5] else 0
            })

        return jsonify({
            "status": "success",
            "count": len(doctors),
            "doctors": doctors
        })
    except Exception as e:
        logger.error(f"Erreur liste médecins : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur de récupération"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@app.route('/appointments', methods=['GET'])
def get_appointments():
    """Récupère les rendez-vous d'un utilisateur"""
    conn = None
    cur = None
    try:
        api_key = request.args.get('api_key')
        if not api_key:
            return jsonify({"status": "error", "message": "Clé API requise"}), 400

        if not validate_api_key(api_key):
            return jsonify({"status": "error", "message": "Clé API invalide"}), 403

        conn = get_db()
        cur = conn.cursor()

        # Récupérer l'ID et le rôle de l'utilisateur
        cur.execute("SELECT id, role FROM users WHERE api_key = %s", (api_key,))
        user = cur.fetchone()
        if not user:
            return jsonify({"status": "error", "message": "Utilisateur non trouvé"}), 404

        user_id, role = user

        if role == 'patient':
            cur.execute("""
                SELECT a.id, a.appointment_date, a.duration, a.status, a.notes,
                       u.username as doctor_name, d.specialties
                FROM appointments a
                JOIN doctors d ON a.doctor_id = d.id
                JOIN users u ON d.user_id = u.id
                WHERE a.patient_id = %s
                ORDER BY a.appointment_date DESC
            """, (user_id,))
        else:  # Médecin
            cur.execute("""
                SELECT a.id, a.appointment_date, a.duration, a.status, a.notes,
                       u.username as patient_name
                FROM appointments a
                JOIN users u ON a.patient_id = u.id
                WHERE a.doctor_id = (SELECT id FROM doctors WHERE user_id = %s)
                ORDER BY a.appointment_date DESC
            """, (user_id,))

        appointments = []
        for row in cur.fetchall():
            appointments.append({
                "id": row[0],
                "date": row[1].isoformat(),
                "duration": row[2],
                "status": row[3],
                "notes": row[4],
                "with_user": row[5],  # Nom du médecin ou patient selon le rôle
                "specialties": json.loads(row[6]) if len(row) > 6 and row[6] else None
            })

        return jsonify({
            "status": "success",
            "appointments": appointments,
            "role": role
        })
    except Exception as e:
        logger.error(f"Erreur rendez-vous : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur de récupération"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@app.route('/appointments', methods=['POST'])
def create_appointment():
    """Crée un nouveau rendez-vous"""
    conn = None
    cur = None
    try:
        data = request.get_json()
        required = ['api_key', 'doctor_id', 'appointment_date']
        if not all(k in data for k in required):
            return jsonify({"status": "error", "message": "Champs manquants"}), 400

        if not validate_api_key(data['api_key']):
            return jsonify({"status": "error", "message": "Clé API invalide"}), 403

        appointment_date = datetime.fromisoformat(data['appointment_date'])
        duration = data.get('duration', 30)

        conn = get_db()
        cur = conn.cursor()

        # Vérifier si le créneau est disponible
        cur.execute("""
            SELECT 1 FROM appointments 
            WHERE doctor_id = %s 
            AND appointment_date = %s
            AND status != 'canceled'
        """, (data['doctor_id'], appointment_date))

        if cur.fetchone():
            return jsonify({"status": "error", "message": "Créneau indisponible"}), 409

        # Créer le rendez-vous
        cur.execute("""
            INSERT INTO appointments (
                patient_id, doctor_id, appointment_date, duration, notes
            ) VALUES (
                (SELECT id FROM users WHERE api_key = %s),
                %s, %s, %s, %s
            ) RETURNING id
        """, (
            data['api_key'],
            data['doctor_id'],
            appointment_date,
            duration,
            data.get('notes', '')
        ))

        appointment_id = cur.fetchone()[0]
        conn.commit()

        # Récupérer les infos pour la notification
        cur.execute("""
            SELECT u.username, d.user_id 
            FROM doctors d
            JOIN users u ON d.user_id = u.id
            WHERE d.id = %s
        """, (data['doctor_id'],))
        doctor_info = cur.fetchone()

        # Envoyer des notifications
        patient_id = get_user_id_from_api_key(data['api_key'])
        if patient_id:
            create_notification(
                patient_id,
                "Rendez-vous confirmé",
                f"Rendez-vous avec Dr. {doctor_info[0]} le {appointment_date.strftime('%d/%m/%Y à %H:%M')}",
                "appointment",
                {"appointment_id": appointment_id}
            )

        if doctor_info[1]:  # ID utilisateur du médecin
            create_notification(
                doctor_info[1],
                "Nouveau rendez-vous",
                f"Rendez-vous le {appointment_date.strftime('%d/%m/%Y à %H:%M')}",
                "appointment",
                {"appointment_id": appointment_id}
            )

        return jsonify({
            "status": "success",
            "appointment_id": appointment_id,
            "message": "Rendez-vous créé"
        }), 201
    except ValueError:
        return jsonify({"status": "error", "message": "Format de date invalide"}), 400
    except Exception as e:
        logger.error(f"Erreur création RDV: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({"status": "error", "message": "Erreur serveur"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@app.route('/appointments/<int:appointment_id>', methods=['PUT'])
def update_appointment(appointment_id):
    """Met à jour un rendez-vous"""
    conn = None
    cur = None
    try:
        data = request.get_json()
        if not data or 'api_key' not in data:
            return jsonify({"status": "error", "message": "Clé API requise"}), 400

        if not validate_api_key(data['api_key']):
            return jsonify({"status": "error", "message": "Clé API invalide"}), 403

        conn = get_db()
        cur = conn.cursor()

        # Vérifier que l'utilisateur a le droit de modifier ce RDV
        cur.execute("""
            SELECT a.patient_id, a.doctor_id, d.user_id as doctor_user_id
            FROM appointments a
            LEFT JOIN doctors d ON a.doctor_id = d.id
            WHERE a.id = %s
        """, (appointment_id,))
        appointment = cur.fetchone()

        if not appointment:
            return jsonify({"status": "error", "message": "Rendez-vous non trouvé"}), 404

        patient_id, doctor_id, doctor_user_id = appointment
        user_id = get_user_id_from_api_key(data['api_key'])

        if user_id not in [patient_id, doctor_user_id]:
            return jsonify({"status": "error", "message": "Non autorisé"}), 403

        # Construire la requête de mise à jour
        updates = []
        params = []

        if 'status' in data:
            updates.append("status = %s")
            params.append(data['status'])

        if 'notes' in data:
            updates.append("notes = %s")
            params.append(data['notes'])

        if 'appointment_date' in data:
            new_date = datetime.fromisoformat(data['appointment_date'])
            # Vérifier que le nouveau créneau est disponible
            cur.execute("""
                SELECT 1 FROM appointments
                WHERE doctor_id = %s
                AND appointment_date = %s
                AND id != %s
                AND status != 'canceled'
            """, (doctor_id, new_date, appointment_id))

            if cur.fetchone():
                return jsonify({"status": "error", "message": "Créneau indisponible"}), 409

            updates.append("appointment_date = %s")
            params.append(new_date)

        if not updates:
            return jsonify({"status": "error", "message": "Aucune modification fournie"}), 400

        params.append(appointment_id)
        query = f"UPDATE appointments SET {', '.join(updates)} WHERE id = %s RETURNING appointment_date, status"

        cur.execute(query, params)
        updated_appointment = cur.fetchone()
        conn.commit()

        # Envoyer des notifications
        if patient_id != user_id:  # Le médecin a modifié le RDV
            create_notification(
                patient_id,
                "Rendez-vous modifié",
                f"Votre rendez-vous a été modifié: {updated_appointment[1]}",
                "appointment",
                {"appointment_id": appointment_id}
            )
        elif doctor_user_id:  # Le patient a modifié le RDV
            create_notification(
                doctor_user_id,
                "Rendez-vous modifié",
                f"Rendez-vous modifié: {updated_appointment[1]}",
                "appointment",
                {"appointment_id": appointment_id}
            )

        return jsonify({
            "status": "success",
            "message": "Rendez-vous mis à jour",
            "new_date": updated_appointment[0].isoformat(),
            "new_status": updated_appointment[1]
        })
    except ValueError:
        return jsonify({"status": "error", "message": "Format de date invalide"}), 400
    except Exception as e:
        logger.error(f"Erreur mise à jour RDV: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({"status": "error", "message": "Erreur serveur"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@app.route('/notifications', methods=['GET'])
def get_notifications():
    """Récupère les notifications d'un utilisateur"""
    conn = None
    cur = None
    try:
        api_key = request.args.get('api_key')
        if not api_key or not validate_api_key(api_key):
            return jsonify({"status": "error", "message": "Non autorisé"}), 403

        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, title, message, is_read, created_at, metadata
            FROM notifications
            WHERE user_id = (SELECT id FROM users WHERE api_key = %s)
            ORDER BY created_at DESC
            LIMIT 50
        """, (api_key,))

        notifications = []
        for row in cur.fetchall():
            notifications.append({
                "id": row[0],
                "title": row[1],
                "message": row[2],
                "is_read": row[3],
                "created_at": row[4].isoformat(),
                "metadata": json.loads(row[5]) if row[5] else None
            })

        return jsonify({
            "status": "success",
            "notifications": notifications
        })
    except Exception as e:
        logger.error(f"Erreur récup. notifications: {str(e)}")
        return jsonify({"status": "error", "message": "Erreur serveur"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@app.route('/notifications/<int:notification_id>', methods=['PUT'])
def mark_notification_read(notification_id):
    """Marque une notification comme lue"""
    conn = None
    cur = None
    try:
        api_key = request.args.get('api_key')
        if not api_key or not validate_api_key(api_key):
            return jsonify({"status": "error", "message": "Non autorisé"}), 403

        conn = get_db()
        cur = conn.cursor()

        # Vérifier que la notification appartient à l'utilisateur
        cur.execute("""
            UPDATE notifications
            SET is_read = TRUE
            WHERE id = %s AND user_id = (SELECT id FROM users WHERE api_key = %s)
            RETURNING id
        """, (notification_id, api_key))

        if not cur.fetchone():
            return jsonify({"status": "error", "message": "Notification non trouvée"}), 404

        conn.commit()
        return jsonify({"status": "success", "message": "Notification marquée comme lue"})
    except Exception as e:
        logger.error(f"Erreur marquage notification: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({"status": "error", "message": "Erreur serveur"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@app.route('/stats/predictions', methods=['GET'])
def get_prediction_stats():
    """Récupère des statistiques sur les prédictions"""
    conn = None
    cur = None
    try:
        conn = get_db()
        cur = conn.cursor()

        # Statistiques globales
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(prediction) as average_risk,
                COUNT(CASE WHEN risk = TRUE THEN 1 END) as high_risk_count
            FROM predictions
        """)
        global_stats = cur.fetchone()

        # Statistiques par jour (7 derniers jours)
        cur.execute("""
            SELECT 
                DATE(timestamp) as day,
                COUNT(*) as count,
                AVG(prediction) as average_risk
            FROM predictions
            WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY day
            ORDER BY day
        """)
        daily_stats = []
        for row in cur.fetchall():
            daily_stats.append({
                "date": row[0].isoformat(),
                "count": row[1],
                "average_risk": float(row[2]) if row[2] else 0
            })

        return jsonify({
            "status": "success",
            "stats": {
                "total_predictions": global_stats[0],
                "average_risk": round(float(global_stats[1]), 2) if global_stats[1] else 0,
                "high_risk_percentage": round((global_stats[2] / global_stats[0] * 100), 2) if global_stats[
                                                                                                   0] > 0 else 0,
                "daily_stats": daily_stats
            }
        })
    except Exception as e:
        logger.error(f"Erreur stats prédictions: {str(e)}")
        return jsonify({"status": "error", "message": "Erreur serveur"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# --- WebSocket ---
@socketio.on('connect')
def handle_connect():
    api_key = request.args.get('api_key')
    if api_key and validate_api_key(api_key):
        user_id = get_user_id_from_api_key(api_key)
        if user_id:
            emit('connection', {'status': 'connected', 'user_id': user_id})
        else:
            emit('connection', {'status': 'unauthorized'})
            disconnect()
    else:
        emit('connection', {'status': 'unauthorized'})
        disconnect()


@socketio.on('join_notifications')
def handle_join_notifications(data):
    api_key = data.get('api_key')
    if api_key and validate_api_key(api_key):
        user_id = get_user_id_from_api_key(api_key)
        if user_id:
            join_room(f'user_{user_id}')
            emit('notification_status', {'status': 'joined'})


@app.route('/stats/users', methods=['GET'])
def get_user_stats():
    """Statistiques sur les utilisateurs"""
    conn = None
    cur = None
    try:
        conn = get_db()
        cur = conn.cursor()

        # Statistiques globales
        cur.execute("""
            SELECT 
                COUNT(*) as total_users,
                COUNT(CASE WHEN role = 'doctor' THEN 1 END) as doctors,
                COUNT(CASE WHEN role = 'patient' THEN 1 END) as patients,
                DATE(MIN(created_at)) as first_signup,
                COUNT(DISTINCT DATE(created_at)) as active_days
            FROM users
        """)
        stats = cur.fetchone()

        # Inscription par mois
        cur.execute("""
            SELECT 
                DATE_TRUNC('month', created_at) as month,
                COUNT(*) as new_users
            FROM users
            GROUP BY month
            ORDER BY month
        """)
        monthly_stats = []
        for row in cur.fetchall():
            monthly_stats.append({
                "month": row[0].strftime("%Y-%m"),
                "new_users": row[1]
            })

        return jsonify({
            "status": "success",
            "stats": {
                "total_users": stats[0],
                "doctors": stats[1],
                "patients": stats[2],
                "first_signup": stats[3].isoformat() if stats[3] else None,
                "active_days": stats[4],
                "monthly_growth": monthly_stats
            }
        })
    except Exception as e:
        logger.error(f"Erreur stats utilisateurs: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@app.route('/stats/activity', methods=['GET'])
def get_activity_stats():
    """Statistiques d'activité"""
    conn = None
    cur = None
    try:
        conn = get_db()
        cur = conn.cursor()

        # Activité récente
        cur.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(DISTINCT user_id) as active_users,
                MAX(timestamp) as last_activity
            FROM predictions
        """)
        activity = cur.fetchone()

        # Prédictions par heure de la journée
        cur.execute("""
            SELECT 
                EXTRACT(HOUR FROM timestamp) as hour,
                COUNT(*) as prediction_count
            FROM predictions
            GROUP BY hour
            ORDER BY hour
        """)
        hourly_activity = []
        for row in cur.fetchall():
            hourly_activity.append({
                "hour": int(row[0]),
                "count": row[1]
            })

        return jsonify({
            "status": "success",
            "activity": {
                "total_predictions": activity[0],
                "active_users": activity[1],
                "last_activity": activity[2].isoformat() if activity[2] else None,
                "hourly_activity": hourly_activity
            }
        })
    except Exception as e:
        logger.error(f"Erreur stats activité: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# ===== Email Provider (Connexion & Reset Password) =====

def send_email(to_email, subject, html_content):
    """Fonction utilitaire pour envoyer des emails"""
    # message = Mail(
    #     from_email=app.config['SENDER_EMAIL'],
    #     to_emails=to_email,
    #     subject=subject,
    #     html_content=html_content)

    message="hello worl"

    try:
        response = sg.send(message)
        logger.info(f"Email envoyé à {to_email}, status: {response.status_code}")
        return True
    except Exception as e:
        logger.error(f"Erreur envoi email: {str(e)}")
        return False


@app.route('/request-password-reset', methods=['POST'])
def request_password_reset():
    """Demande de réinitialisation de mot de passe"""
    conn = None
    cur = None
    try:
        email = request.json.get('email')
        if not email:
            return jsonify({"status": "error", "message": "Email requis"}), 400

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id, username FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

        if user:
            user_id, username = user
            # Générer un token valide 24h
            token = ts.dumps(user_id, salt=app.config['SECURITY_PASSWORD_SALT'])

            reset_url = f"{app.config['FRONTEND_URL']}/reset-password?token={token}"

            # Envoyer l'email
            html_content = f"""
                <h2>Réinitialisation de mot de passe</h2>
                <p>Bonjour {username},</p>
                <p>Vous avez demandé à réinitialiser votre mot de passe. Cliquez sur le lien ci-dessous :</p>
                <p><a href="{reset_url}">Réinitialiser mon mot de passe</a></p>
                <p>Ce lien expirera dans 24 heures.</p>
                <p>Si vous n'avez pas fait cette demande, ignorez simplement cet email.</p>
            """

            if send_email(email, "Réinitialisation de mot de passe", html_content):
                return jsonify({"status": "success", "message": "Email envoyé"})
            else:
                return jsonify({"status": "error", "message": "Erreur d'envoi d'email"}), 500

        # Pour des raisons de sécurité, on ne révèle pas si l'email existe
        return jsonify({"status": "success", "message": "Si l'email existe, un lien de réinitialisation a été envoyé"})
    except Exception as e:
        logger.error(f"Erreur demande reset password: {str(e)}")
        return jsonify({"status": "error", "message": "Erreur serveur"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@app.route('/reset-password', methods=['POST'])
def reset_password():
    """Réinitialisation du mot de passe avec token"""
    conn = None
    cur = None
    try:
        token = request.json.get('token')
        new_password = request.json.get('new_password')

        if not token or not new_password:
            return jsonify({"status": "error", "message": "Token et nouveau mot de passe requis"}), 400

        # Vérifier le token
        try:
            user_id = ts.loads(
                token,
                salt=app.config['SECURITY_PASSWORD_SALT'],
                max_age=86400  # 24 heures
            )
        except:
            return jsonify({"status": "error", "message": "Token invalide ou expiré"}), 400

        # Mettre à jour le mot de passe
        password_hash = generate_password_hash(new_password)

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s RETURNING email, username",
            (password_hash, user_id)
        )
        user = cur.fetchone()

        if not user:
            return jsonify({"status": "error", "message": "Utilisateur non trouvé"}), 404

        conn.commit()

        # Envoyer une confirmation par email
        email, username = user
        html_content = f"""
            <h2>Mot de passe mis à jour</h2>
            <p>Bonjour {username},</p>
            <p>Votre mot de passe a été modifié avec succès.</p>
            <p>Si vous n'avez pas effectué cette modification, veuillez nous contacter immédiatement.</p>
        """
        send_email(email, "Confirmation de changement de mot de passe", html_content)

        return jsonify({"status": "success", "message": "Mot de passe mis à jour"})
    except Exception as e:
        logger.error(f"Erreur reset password: {str(e)}")
        if conn:
            conn.rollback()
        return jsonify({"status": "error", "message": "Erreur serveur"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# --- Point d'entrée ---
# if __name__ == '__main__':
#     try:
#         init_db()
#         socketio.run(app, host='0.0.0.0', port=5000, debug=True)
#     except Exception as e:
#         logger.error(f"Application startup failed: {str(e)}")
#         raise

if __name__ == '__main__':
    try:
        init_db()
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise








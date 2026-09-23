from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_file,
)

import joblib
import pandas as pd
import sqlite3
import io

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

app.secret_key = "change-this-secret-key"

DATABASE = "fraud_detection.db"

MODEL_PATH = "model/fraud_model.pkl"

model = joblib.load(MODEL_PATH)


FEATURES = [
    "transaction_amount",
    "account_age_days",
    "previous_transactions",
    "failed_transactions",
    "transaction_hour",
    "is_international",
    "device_change",
]


# ==========================================
# DATABASE
# ==========================================


def get_db_connection():

    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_db_connection()

    # Users table

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL,

            created_at TEXT

        )
    """)

    # Transactions table

    connection.execute("""
        CREATE TABLE IF NOT EXISTS transactions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER,

            transaction_amount REAL,

            account_age_days INTEGER,

            previous_transactions INTEGER,

            failed_transactions INTEGER,

            transaction_hour INTEGER,

            is_international INTEGER,

            device_change INTEGER,

            prediction TEXT,

            risk_score REAL,

            risk_level TEXT,

            created_at TEXT,

            FOREIGN KEY(user_id)
            REFERENCES users(id)

        )
    """)

    connection.commit()

    connection.close()


# ==========================================
# LOGIN REQUIRED
# ==========================================


def login_required():

    return "user_id" in session


# ==========================================
# REGISTER
# ==========================================


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"].strip()

        email = request.form["email"].strip().lower()

        password = request.form["password"]

        if not name or not email or not password:

            flash("All fields are required.", "error")

            return redirect(url_for("register"))

        if len(password) < 6:

            flash("Password must contain at least 6 characters.", "error")

            return redirect(url_for("register"))

        connection = get_db_connection()

        existing_user = connection.execute(
            """

            SELECT id

            FROM users

            WHERE email = ?

        """,
            (email,),
        ).fetchone()

        if existing_user:

            connection.close()

            flash("Email already registered.", "error")

            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        connection.execute(
            """

            INSERT INTO users (

                name,
                email,
                password,
                created_at

            )

            VALUES (?, ?, ?, ?)

        """,
            (
                name,
                email,
                hashed_password,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

        connection.commit()

        connection.close()

        flash("Account created successfully. Please login.", "success")

        return redirect(url_for("login"))

    return render_template("register.html")


# ==========================================
# LOGIN
# ==========================================


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"].strip().lower()

        password = request.form["password"]

        connection = get_db_connection()

        user = connection.execute(
            """

            SELECT *

            FROM users

            WHERE email = ?

        """,
            (email,),
        ).fetchone()

        connection.close()

        if user and check_password_hash(user["password"], password):

            session["user_id"] = user["id"]

            session["user_name"] = user["name"]

            session["user_email"] = user["email"]

            return redirect(url_for("home"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


# ==========================================
# LOGOUT
# ==========================================


@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# ==========================================
# DASHBOARD
# ==========================================


@app.route("/")
def home():

    if not login_required():

        return redirect(url_for("login"))

    user_id = session["user_id"]

    connection = get_db_connection()

    transactions = connection.execute(
        """

        SELECT *

        FROM transactions

        WHERE user_id = ?

        ORDER BY id DESC

        LIMIT 10

    """,
        (user_id,),
    ).fetchall()

    total_transactions = connection.execute(
        """

        SELECT COUNT(*) AS count

        FROM transactions

        WHERE user_id = ?

    """,
        (user_id,),
    ).fetchone()["count"]

    fraud_transactions = connection.execute(
        """

        SELECT COUNT(*) AS count

        FROM transactions

        WHERE user_id = ?

        AND prediction = 'FRAUD'

    """,
        (user_id,),
    ).fetchone()["count"]

    genuine_transactions = connection.execute(
        """

        SELECT COUNT(*) AS count

        FROM transactions

        WHERE user_id = ?

        AND prediction = 'GENUINE'

    """,
        (user_id,),
    ).fetchone()["count"]

    connection.close()

    if total_transactions > 0:

        fraud_rate = round((fraud_transactions / total_transactions) * 100, 2)

    else:

        fraud_rate = 0

    return render_template(
        "index.html",
        transactions=transactions,
        total_transactions=total_transactions,
        fraud_transactions=fraud_transactions,
        genuine_transactions=genuine_transactions,
        fraud_rate=fraud_rate,
        user_name=session["user_name"],
    )


# ==========================================
# SINGLE PREDICTION
# ==========================================


@app.route("/predict", methods=["POST"])
def predict():

    if not login_required():

        return redirect(url_for("login"))

    try:

        transaction_amount = float(request.form["transaction_amount"])

        account_age_days = int(request.form["account_age_days"])

        previous_transactions = int(request.form["previous_transactions"])

        failed_transactions = int(request.form["failed_transactions"])

        transaction_hour = int(request.form["transaction_hour"])

        is_international = int(request.form["is_international"])

        device_change = int(request.form["device_change"])

        input_data = pd.DataFrame(
            [
                {
                    "transaction_amount": transaction_amount,
                    "account_age_days": account_age_days,
                    "previous_transactions": previous_transactions,
                    "failed_transactions": failed_transactions,
                    "transaction_hour": transaction_hour,
                    "is_international": is_international,
                    "device_change": device_change,
                }
            ]
        )

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        risk_score = round(probability * 100, 2)

        if risk_score >= 70:

            risk_level = "HIGH"

        elif risk_score >= 40:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        result = "FRAUD" if prediction == 1 else "GENUINE"

        connection = get_db_connection()

        connection.execute(
            """

            INSERT INTO transactions (

                user_id,

                transaction_amount,

                account_age_days,

                previous_transactions,

                failed_transactions,

                transaction_hour,

                is_international,

                device_change,

                prediction,

                risk_score,

                risk_level,

                created_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """,
            (
                session["user_id"],
                transaction_amount,
                account_age_days,
                previous_transactions,
                failed_transactions,
                transaction_hour,
                is_international,
                device_change,
                result,
                risk_score,
                risk_level,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

        connection.commit()

        connection.close()

        return render_template(
            "result.html",
            result=result,
            risk_score=risk_score,
            risk_level=risk_level,
            transaction_amount=transaction_amount,
        )

    except Exception as e:

        return f"Error: {str(e)}"


# ==========================================
# ALL TRANSACTIONS
# ==========================================


@app.route("/transactions")
def transactions():

    if not login_required():

        return redirect(url_for("login"))

    search = request.args.get("search", "").strip()

    prediction_filter = request.args.get("prediction", "")

    risk_filter = request.args.get("risk", "")

    query = """

        SELECT *

        FROM transactions

        WHERE user_id = ?

    """

    params = [session["user_id"]]

    if search:

        query += """

            AND (

                CAST(transaction_amount AS TEXT)
                LIKE ?

                OR prediction LIKE ?

            )

        """

        search_value = f"%{search}%"

        params.extend([search_value, search_value])

    if prediction_filter:

        query += """
            AND prediction = ?
        """

        params.append(prediction_filter)

    if risk_filter:

        query += """
            AND risk_level = ?
        """

        params.append(risk_filter)

    query += """

        ORDER BY id DESC

    """

    connection = get_db_connection()

    results = connection.execute(query, params).fetchall()

    connection.close()

    return render_template(
        "transactions.html",
        transactions=results,
        search=search,
        prediction_filter=prediction_filter,
        risk_filter=risk_filter,
    )


# ==========================================
# DELETE TRANSACTION
# ==========================================


@app.route("/delete-transaction/<int:transaction_id>", methods=["POST"])
def delete_transaction(transaction_id):

    if not login_required():

        return redirect(url_for("login"))

    connection = get_db_connection()

    connection.execute(
        """

        DELETE FROM transactions

        WHERE id = ?

        AND user_id = ?

    """,
        (transaction_id, session["user_id"]),
    )

    connection.commit()

    connection.close()

    return redirect(url_for("transactions"))


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    init_database()

    app.run(debug=True)

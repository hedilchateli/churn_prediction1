import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def prepare_data(file_path_1, file_path_2):
    """Charge et prétraite les données depuis deux fichiers CSV."""
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    # Concaténer les deux datasets
    df = pd.concat([df1, df2], ignore_index=True)

    # Vérifier si la colonne cible est la dernière (modifie si nécessaire)
    target_column = df.columns[-1]  # Dernière colonne comme cible
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encodage des variables catégoriques en nombres
    X = pd.get_dummies(
        X
    )  # Convertit les variables catégoriques en indicateurs numériques

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Entraîne un modèle de régression logistique."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et affiche la précision."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def save_model(model, filename="model.pkl"):
    """Sauvegarde le modèle entraîné."""
    joblib.dump(model, filename)


def load_model(filename="model.pkl"):
    """Charge un modèle sauvegardé."""
    return joblib.load(filename)

import mlflow
import mlflow.sklearn
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    # Initialiser l'expérience MLflow
    mlflow.set_experiment("Churn_Prediction")

    # Démarrer un run MLflow
    with mlflow.start_run():
        # Charger et préparer les données
        file_path_1 = "churn-bigml-80.csv"
        file_path_2 = "churn-bigml-20.csv"
        X_train, X_test, y_train, y_test = prepare_data(file_path_1, file_path_2)

        # Logger les paramètres des données
        mlflow.log_param("training_data", file_path_1)
        mlflow.log_param("testing_data", file_path_2)

        # Entraîner le modèle
        model = train_model(X_train, y_train)

        # Évaluer le modèle
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Précision du modèle : {accuracy:.2f}")

        # Logger la métrique
        mlflow.log_metric("accuracy", accuracy)

        # Sauvegarder le modèle
        save_model(model)

        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(model, "churn_model")

        # Charger et tester le modèle sauvegardé
        loaded_model = load_model()
        loaded_accuracy = evaluate_model(loaded_model, X_test, y_test)
        print(f"Précision après chargement : {loaded_accuracy:.2f}")

        # Logger la précision après chargement
        mlflow.log_metric("loaded_model_accuracy", loaded_accuracy)


if __name__ == "__main__":
    main()


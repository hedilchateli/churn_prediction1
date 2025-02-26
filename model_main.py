from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    # Charger et préparer les données depuis les deux fichiers
    file_path_1 = "churn-bigml-80.csv"
    file_path_2 = "churn-bigml-20.csv"

    X_train, X_test, y_train, y_test = prepare_data(file_path_1, file_path_2)

    # Entraîner le modèle
    model = train_model(X_train, y_train)

    # Évaluer le modèle
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Précision du modèle : {accuracy:.2f}")

    # Sauvegarder le modèle
    save_model(model)

    # Charger et tester le modèle sauvegardé
    loaded_model = load_model()
    loaded_accuracy = evaluate_model(loaded_model, X_test, y_test)
    print(f"Précision après chargement : {loaded_accuracy:.2f}")


if __name__ == "__main__":
    main()

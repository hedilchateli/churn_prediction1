# Étape 01 : Préparer l’environnement
.PHONY: venv install

venv:
	@echo "Création d'un environnement virtuel..."
	python3 -m venv venv

install: venv
	@echo "Installation des dépendances..."
	./venv/bin/pip install -r requirements.txt

# Étape 02 : Définir les sections du Makefile

# Vérification du code
.PHONY: format lint security
format:
	@echo "Formatage du code..."
	./venv/bin/black .

lint:
	@echo "Vérification de la qualité du code..."
	./venv/bin/flake8 .

security:
	@echo "Vérification de la sécurité du code..."
	./venv/bin/bandit -r .

# Exécution des tests unitaires
.PHONY: test
test:
	@echo "Exécution du test..."
	 python3 main.py test

# Mise à jour des dépendances
.PHONY: update
update:
	@echo "Mise à jour des dépendances..."
	./venv/bin/pip install --upgrade -r requirements.txt

# Génération de la documentation
.PHONY: docs
docs:
	@echo "Génération de la documentation..."
	./venv/bin/sphinx-build -b html docs/source docs/build

# Nettoyage des fichiers temporaires
.PHONY: clean
clean:
	@echo "Nettoyage des fichiers temporaires..."
	rm -rf *.pyc __pycache__ .pytest_cache .tox .coverage .nox

# Vérification de la couverture du code
.PHONY: coverage
coverage:
	@echo "Vérification de la couverture du code..."
	./venv/bin/pytest --cov=./ --cov-report=html

# Préparer les données
.PHONY: prepare_data
prepare_data:
	@echo "Préparation des données..."
	python3 main.py prepare_data


# Installer les dépendances à partir de requirements.txt
install_requirements:
	@echo "Installation des dépendances..."
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
check_venv:
	@echo "Vérification de l'environnement virtuel..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
	   echo "Erreur : L'environnement virtuel n'est pas activé."; \
	   exit 1; \
	else \
	   echo "L'environnement virtuel est actif."; \
	fi

run_mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

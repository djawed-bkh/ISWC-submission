# Temporal Constraints - QCN Generator

Système d'apprentissage de contraintes temporelles basé sur les relations d'Allen pour les knowledge graphs Wikidata.

## 📋 Prérequis

### Dépendances Python
```bash
uv venv .venv
uv sync --dev
```

### Configuration du dossier de données
Par défaut, les jeux de données sont lus depuis `data/`.

Vous pouvez surcharger ce chemin avec la variable d'environnement `TCLKG_DATA_DIR`:

```bash
# chemin absolu
export TCLKG_DATA_DIR=/path/to/data

# ou chemin relatif à la racine du projet
export TCLKG_DATA_DIR=custom-data
```

### Structure de données requise
Le projet nécessite des fichiers de données au format quintuplet dans la structure suivante:
```
data/
  └── {KG_type}/
      └── train_cst_knowledge.quintuplet
```

**Format du fichier quintuplet** (séparateur: tabulation):
```
<head_uri>	<relation_uri>	<value>	<start_date>	<end_date>
```

Exemple:
```
http://www.wikidata.org/entity/Q142	http://www.wikidata.org/prop/P6	http://www.wikidata.org/entity/Q191954	1959-01-08	None
```

## 🚀 Exécution

### Lancer l'apprentissage de contraintes
```bash
uv run python scripts/qcn_generator.py Q6256 0.8 600
```

Entrypoints disponibles:
- `scripts/qcn_generator.py`
- `scripts/qcn_generator2.py`
- `scripts/qcn_generator_unmax.py`
- `scripts/perf_smoke.py`
- `scripts/run_experiments.py`

## ✅ Qualité & CI

### Checks CI automatiques
Le workflow GitHub Actions (`.github/workflows/ci.yml`) exécute sur `push` et `pull_request`:
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pyright`
- `uv run pytest`
- `uv run coverage run -m pytest` puis `uv run coverage report -m`

### Couverture de tests
La couverture est configurée dans `pyproject.toml`:
- périmètre: `src/tclkg`
- couverture de branches activée
- seuil minimal global: `fail_under = 45`

### Smoke performance
Commande reproductible pour détecter les régressions de performance:

```bash
uv run python scripts/perf_smoke.py --iterations 5 --max-total-ms 120
```

Alternative via entrypoint:

```bash
uv run tclkg-perf-smoke --iterations 5 --max-total-ms 120
```

Le code de sortie est `1` si `total_ms` dépasse `--max-total-ms`.

### Pre-commit local
La configuration `.pre-commit-config.yaml` reflète les checks CI. Installation et exécution:

```bash
uv run pre-commit install
uv run pre-commit run -a
```

## 🗂️ Structure du projet

```text
src/tclkg/         # package principal (code métier)
scripts/           # entrypoints CLI
data/              # jeux de données
Results/           # sorties d'expériences
```

### Configuration des paramètres

Les paramètres principaux se passent en arguments CLI:

```bash
uv run python scripts/qcn_generator.py <kg> <threshold> <timeout_seconds>
```

**Paramètres disponibles:**
- `kg`: Liste des types d'entités Wikidata (QID)
  - `Q6256`: Pays
  - `Q215380`: Groupes musicaux
  - `Q82955`: Villes (si données disponibles)

- `percentage_entity`: Seuil minimal d'entités pour filtrer les propriétés multivaluées (en %)
  - `10`: Filtre agressif (plus de propriétés supprimées)
  - `5`: Filtre modéré
  - `2`: Filtre léger (moins de propriétés supprimées)

- `threshold`: Seuil de confiance pour la validation des relations d'Allen (0 à 1)
  - `0.95`: Très strict (95% de correspondance requise)
  - `0.9`: Strict (90% de correspondance)
  - `0.8`: Modéré (80% de correspondance)

- `timeout`: Durée maximale d'apprentissage en secondes (défaut: 1200 = 20 minutes)

## 📦 Modules utilisés

### Modules locaux requis
- `src/tclkg/time_package.py` - Classes de base (Entity, Triple, Interval)
- `src/tclkg/allen_relations.py` - Vérification des relations d'Allen
- `src/tclkg/allen_list.py` - Définitions et compositions des relations d'Allen
- `src/tclkg/kg_helpers.py` - Utilitaires pour manipuler les quintuplets
- `src/tclkg/rule_discovery.py` - Fonctions de découverte de règles et multivaluation

### Architecture des relations d'Allen

Le système implémente les 13 relations d'Allen:
- **Temporelles**: before, after, meets, met_by
- **Chevauchement**: overlaps, overlapped_by
- **Inclusion**: during, contains
- **Alignement**: starts, started_by, finishes, finished_by
- **Égalité**: equals

## 🔄 Processus d'apprentissage

1. **Chargement du KG**: Lecture du fichier `train_cst_knowledge.quintuplet`
2. **Filtrage multivaluation**: Suppression des propriétés temporellement multivaluées
3. **Construction du réseau**: Création d'un réseau de contraintes par entité
4. **Path Consistency**: Application de la propagation de contraintes avec composition d'Allen
5. **Apprentissage**: Raffinement itératif des domaines de contraintes avec validation oracle

## 📊 Sortie

Le programme affiche:
- Nombre d'entités chargées
- Nombre de propriétés multivaluées détectées
- Nombre d'entités supprimées
- Progression de l'apprentissage
- Réseau de contraintes final par entité

## 🐛 Dépannage

### Erreur "Entity not found"
Vérifiez que le fichier `train_cst_knowledge.quintuplet` existe et contient des données valides.

### Timeout
Augmentez la valeur du paramètre `timeout` dans l'appel à `learner_with_timeout()`.

### Memory Error
Réduisez le nombre d'entités en utilisant un `percentage_entity` plus élevé ou en filtrant le fichier de données.

## 📝 Notes

- **Granularité temporelle**: Fixée à "D" (jour) dans le code
- **Date de référence**: 2023-12-31
- **Format des URIs**: Les URIs Wikidata sont automatiquement normalisées
- **Valeurs None**: Les dates de début/fin peuvent être `None` (intervalles ouverts)

## 🔗 Références

- [Allen's Interval Algebra](https://en.wikipedia.org/wiki/Allen%27s_interval_algebra)
- [Wikidata](https://www.wikidata.org/)

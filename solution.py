import warnings
import pandas as pd
import numpy as np
from utils import get_dataframe, get_metrics
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm
import optuna




# Constantes de configuration :
SEED = 42

NB_JOBS = -1


# Hyperparamètres (Optuna) :
NB_TRIALS = 100

NB_HOLED_TARGETS_FOR_OPTIMIZATION = 25


# Hyperparamètres (imputation finale) :
NB_NEIGHBORS = 500 # BASE_VALUE = 500

RIDGE_PROPORTION = 0.125 # BASE_VALUE = 0.125

RIDGE_PARAMETERS = {
    'alpha': 10, # BASE_VALUE = 10.0
    'random_state': SEED
}
HGBR_PARAMETERS = {
    'max_iter': 250, # BASE_VALUE = 250
    'max_depth': 8, # BASE_VALUE = 8
    'learning_rate': 0.15, # BASE_VALUE = 0.15
    'early_stopping': True,
    'random_state': SEED
}


def ensemble_impute_ridge_HGBR(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_missing: pd.DataFrame,
        ridge_parameters: dict = RIDGE_PARAMETERS,
        hgbr_parameters: dict = HGBR_PARAMETERS,
        ridge_proportion: float = RIDGE_PROPORTION
    ) -> np.ndarray:
    """
    Entraîne les deux modèles donnés ci-dessous et retourne une prédiction combinée, pondérée selon RIDGE_PROPORTION.
        1. Ridge (Linéaire), nécessite une standardisation des données.
        2. HistGradientBoostingRegressor (Non-Linéaire).

    Args:
        x_train (pd.DataFrame): Données d'entraînement (voisins complètes).
        y_train (pd.Series): Cible d'entraînement (valeurs complètes).
        x_missing (pd.DataFrame): Données à prédire (voisins avec trous).
        ridge_parameters (dict): Dictionnaire des hyperparamètres pour le modèle Ridge.
        hgbr_parameters (dict): Dictionnaire des hyperparamètres pour le modèle HGBR.
        ridge_proportion (float): Poids pour la prédiction Ridge dans l'ensemble.
    Returns:
        np.ndarray: Prédictions combinées pour les données manquantes.
    """
    # Création des scalers pour la standardisation des données :
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_missing_scaled = scaler.transform(x_missing)

    # Création et entraînement des modèles :
        # 1. Ridge (Linéaire) :
    ridge_model = Ridge(**ridge_parameters)
    ridge_model.fit(x_train_scaled, y_train)
    ridge_prediction = ridge_model.predict(x_missing_scaled)
        # 2. Gradient Boosting (Non-Linéaire) :
    HGBR_model = HistGradientBoostingRegressor(**hgbr_parameters)
    HGBR_model.fit(x_train, y_train)
    HGBR_prediction = HGBR_model.predict(x_missing)

    # Combinaison des prédictions :
    return (ridge_proportion * ridge_prediction) + ((1 - ridge_proportion) * HGBR_prediction)


def evaluate_optimization_target(df_x: pd.DataFrame, df_y: pd.DataFrame, name_column_to_fill: str, parameters: dict, complete_columns: list) -> float:
    """
    Evalue une colonne cible spécifique en utilisant les hyperparamètres donnés.
    Retourne le MAE sur les valeurs manquantes uniquement.

    Args:
        df_x (pd.DataFrame): DataFrame avec les colonnes trouées.
        df_y (pd.DataFrame): DataFrame avec les colonnes complètes (vérité terrain).
        name_column_to_fill (str): Nom de la colonne cible à évaluer.
        parameters (dict): Dictionnaire des hyperparamètres à utiliser.
        complete_columns (list): Liste des noms de colonnes complètes à utiliser comme prédicteurs.
    Returns:
        float: MAE calculée sur les positions initialement manquantes.
    """
    # Suppression des warnings 'inutiles' dans la console :
    warnings.filterwarnings("ignore")

    # Extraction de la colonne cible et création des masques :
    y_true_full = df_y[name_column_to_fill]
    y_input_holed = df_x[name_column_to_fill]
    
    mask_missing = y_input_holed.isna()
    mask_valid = ~mask_missing

    # Si la colonne est déjà complète, on retourne 0.0 comme erreur :
    if not mask_missing.any(): 
        return 0.0

    y_valid = y_input_holed[mask_valid] # Les valeurs connues à imputer.

    # Sélection des colonnes / features candidates "voisines" de la colonne cible :
    X_candidates = df_x.loc[mask_valid, complete_columns]

    # Calcul des corrélations entre les colonnes candidates et la colonne cible :
    corrs = X_candidates.corrwith(y_valid).abs()
    top_k = corrs.nlargest(parameters['nb_neighbors']).index

    # Préparation des données d'entraînement et de prédiction :
    x_train = df_x.loc[mask_valid, top_k]
    y_train = y_valid
    x_missing = df_x.loc[mask_missing, top_k]

    # Calcul des prédictions avec l'ensemble Ridge + HGBR :
    y_predicted = ensemble_impute_ridge_HGBR(
        x_train, y_train, x_missing,
        ridge_parameters = {
            'alpha': parameters['ridge_alpha'],
            'random_state': SEED
        },
        hgbr_parameters = {
            'max_iter': parameters['hgbr_iter'],
            'max_depth': parameters['hgbr_depth'],
            'learning_rate': parameters['hgbr_lr'],
            'early_stopping': True,
            'random_state': SEED
        },
        ridge_proportion = parameters['w_ridge']
    )

    # Calcul du MAE sur les positions initialement manquantes :
    y_true_missing = y_true_full[mask_missing]
    return np.mean(np.abs(y_true_missing - y_predicted)) # Pas exactement la même chose que dans utils.get_metrics, mais BEAUCOUP plus rapide.


def find_optimized_parameters(
        nb_holed_targets_for_optimization: int = NB_HOLED_TARGETS_FOR_OPTIMIZATION,
        nb_trials: int = NB_TRIALS,
        input_file: str = 'data/datasets/x_train.csv',
        output_file: str = 'data/datasets/y_train_true.csv',
        with_parallelism: bool = True,
        nb_jobs: int = NB_JOBS
    ) -> dict[str, float]:
    """
    Utilise Optuna pour trouver les hyperparamètres optimaux pour l'imputation des colonnes trouées.
    Évalue les performances sur un sous-ensemble de colonnes cibles pour accélérer le processus.

    Args:
        nb_holed_targets_for_optimization (int): Nombre de colonnes 'holed_*' à échantillonner pour l'optimisation des hyperparamètres (parmi les 1000 disponibles).
        nb_trials (int): Nombre de trials pour l'optimisation des hyperparamètres avec Optuna.
        input_file (str):
            Chemin vers le fichier CSV d'entrée avec les colonnes trouées.
            Par défaut 'data/datasets/x_train.csv' (x_train.csv sur le site web).
        output_file (str):
            Chemin vers le fichier CSV de vérité terrain avec les colonnes complètes.
            Par défaut 'data/datasets/y_train_true.csv' (y_train.csv sur le site web).
        with_parallelism (bool): Si vrai, utilise joblib.Parallel pour accélérer le processus d'évaluation.
        nb_jobs (int): Nombre de jobs parallèles à utiliser si with_parallelism est vrai.
    Returns:
        dict[str, float]: Dictionnaire des meilleurs hyperparamètres trouvés.
    """
    # Ouverture des fichiers de données :
    print(f"Chargement des données pour l'optimisation des hyperparamètres.")
    df_x = get_dataframe(input_file)
    df_y = get_dataframe(output_file)

    # Sélection des colonnes cibles et prédicteurs :
    holed_to_predict = [
        column for column in df_x.columns if 'holed_' in column
    ]

    # Extration des colonnes d'entrainement (complètes et non-constantes) :
    potential_predictors = [column for column in df_x.columns if 'holed_' not in column]
    stds = df_x[potential_predictors].std()
    complete_columns = stds[stds > 1e-9].index.tolist() # On enlève les colonnes constantes car elles sont (quasiment) inutiles pour Ridge et HGBR.

    # Choix d'un sous-ensemble aléatoire de nb_holed_targets_for_optimization colonnes cibles (parmis les 1000 'holed_*' pour l'optimisation (trop lent sinon) :
    np.random.seed(SEED)
    subset_targets = np.random.choice(holed_to_predict, size=nb_holed_targets_for_optimization, replace=False)

    print(f"\nRecherche des hyperparamètres les plus optimisés ({nb_trials} trials) :\n\t- Nombre de colonnes 'holed_*' échantillonnées\t = {len(subset_targets)}\n\t- Nombre de colonnes prédictrices\t\t = {len(complete_columns)}\n")

    # Définition de la fonction 'objective' d'Optuna :
    def objective(trial: optuna.trial.Trial) -> float:
        parameters = {
            'nb_neighbors': trial.suggest_int('nb_neighbors', 0, 10000),
            'ridge_alpha':  trial.suggest_float('ridge_alpha', 1.0, 500.0, log=True),
            'hgbr_iter':    trial.suggest_int('hgbr_iter', 100, 500),
            'hgbr_depth':   trial.suggest_int('hgbr_depth', 4, 15),
            'hgbr_lr':      trial.suggest_float('hgbr_lr', 0.01, 0.3),
            'w_ridge':      trial.suggest_float('w_ridge', 0.0, 1.0) 
        }

        if with_parallelism:
            scores = Parallel(n_jobs=nb_jobs)(
                delayed(evaluate_optimization_target)(df_x, df_y, column, parameters, complete_columns)
                for column in subset_targets
            )
        else:
            scores = [
                evaluate_optimization_target(df_x, df_y, column, parameters, complete_columns)
                for column in subset_targets
            ]

        return np.mean(scores)

    # Lancement de l'optimisation avec Optuna :
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=nb_trials)

    # Résultat final :
    print("\nOptimisation complète, meilleurs hyperparamètres trouvés :\n" + f"\n\t".join([f'- {key} = {value}' for key, value in study.best_params.items()]))
    return study.best_params


def impute_column(df: pd.DataFrame, name_column_to_fill: str, complete_columns: list, nb_neighbors: int = NB_NEIGHBORS) -> pd.Series:
    """
    Impute les valeurs manquantes dans une colonne cible donnée en utilisant les colonnes complètes spécifiées.
    Utilise une combinaison de Ridge et HistGradientBoostingRegressor pour l'imputation.

    Args:
        df (pd.DataFrame): DataFrame contenant toutes les données.
        name_column_to_fill (str): Nom de la colonne cible à imputer.
        complete_columns (list): Liste des noms de colonnes complètes à utiliser comme prédicteurs.
        nb_neighbors (int): Nombre de voisins les plus corrélés à utiliser pour l'imputation.
    Returns:
        pd.Series: Série contenant la colonne imputée.
    """
    # Suppression des warnings 'inutiles' dans la console :
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(divide='ignore', invalid='ignore')

    # Extraction de la colonne cible et création des masques :
    y_full_input = df[name_column_to_fill]

    mask_missing = y_full_input.isna()
    mask_valid = ~mask_missing

    # Si la colonne est déjà complète, on la retourne telle quelle :
    if not mask_missing.any():
        return y_full_input

    y_valid = y_full_input[mask_valid] # Les valeurs connues à imputer.

    # Sélection des colonnes / features candidates "voisines" de la colonne cible :
    x_candidates = df.loc[mask_valid, complete_columns]

    # Calcul des corrélations entre les colonnes candidates et la colonne cible :
    correlations = x_candidates.corrwith(y_valid)
    correlations = correlations.dropna()

    # Si aucune corrélation n'est trouvée, on remplit avec la moyenne des valeurs connues :
    if correlations.empty:
        fill_value = y_valid.mean()
        y_completed = y_full_input.fillna(fill_value)

    # Sinon, on procède à l'imputation en utilisant les nb_neighbors voisins les plus corrélés :
    else:
        # Sélection des nb_neighbors voisins les plus corrélés (en valeur absolue) :
        top_neighbors = correlations.abs().nlargest(nb_neighbors).index

        # Préparation des données d'entraînement et de prédiction :
        x_train = df.loc[mask_valid, top_neighbors]
        y_train = y_valid
        x_missing = df.loc[mask_missing, top_neighbors]

        # Prédiction des valeurs manquantes :
        y_predicted = ensemble_impute_ridge_HGBR(x_train, y_train, x_missing)

        # Remplissage des valeurs manquantes dans la série finale :
        y_completed = y_full_input.copy()
        y_completed.loc[mask_missing] = y_predicted

    return y_completed


def impute_holed_columns(df: pd.DataFrame, columns_to_fill: list[str], complete_columns: list[str], with_parallelism: bool = True, nb_jobs: int = NB_JOBS) -> pd.DataFrame:
    """
    Impute les colonnes spécifiées dans 'columns_to_fill' en utilisant les colonnes complètes données dans 'complete_columns'.
    Utilise le parallélisme pour (très largement) accélérer le processus.

    Args:
        df (pd.DataFrame): DataFrame contenant toutes les données.
        columns_to_fill (list): Liste des noms de colonnes à imputer.
        complete_columns (list): Liste des noms de colonnes complètes à utiliser comme prédicteurs.
        with_parallelism (bool): Si vrai, utilise joblib.Parallel pour exécuter les imputations en parallèle.
        nb_jobs (int): Nombre de jobs parallèles à utiliser si with_parallelism est vrai.
    Returns:
        pd.DataFrame: DataFrame contenant les colonnes imputées.
    """
    # Création et exécution des tâches parallèles :
    if with_parallelism:
        results = Parallel(n_jobs=nb_jobs)(
            delayed(impute_column)(df, column, complete_columns)
            for column in tqdm(columns_to_fill)
        )
    else :
        results = [
            impute_column(df, column, complete_columns)
            for column in tqdm(columns_to_fill)
        ]

    # Reconstruction du DataFrame final :
        # Convertion du résultat en un dictionnaire de (column_name, series) :
    result_dict: dict[str, pd.Series] = dict(zip(columns_to_fill, results))
        # Concaténation de toutes les séries prédites :
    result_df = pd.concat(result_dict, axis=1)
        # On restaure l'index d'origine (Horodate) afin de pouvoir exporter le résuktats en un CSV valide :
    result_df.index.name = 'Horodate'
    result_df.reset_index(inplace=True)
        # Reconstruction des colonnes, dans le bon ordre :
    columns_to_keep = ['Horodate'] + [column for column in result_df.columns if 'holed_' in column]

    return result_df[columns_to_keep]


def run_imputation(train_mode: bool = False, input_file: str = None, output_file: str = None) -> None:
    """
    Fonction principale pour exécuter l'imputation des colonnes trouées dans un fichier donné.
    Charge les données, effectue l'imputation et sauvegarde les résultats dans un fichier CSV.

    Args:
        train_mode (bool):
            Si vrai, réalise une prédiction sur le dataset de train et renvoie le MAE & R² associée.
            Si faux, réalise une prédiction sur le dataset de test.
        input_file (str):
            Chemin vers le fichier d'entrée contenant les données avec trous.
            Par défaut 'data/datasets/x_train.csv' si train_mode == True, sinon 'data/datasets/x_test.csv'.
        output_file (str):
            Chemin vers le fichier de sortie pour enregistrer les données imputées.
            Par défaut 'y_train.csv' si train_mode == True, sinon 'y_test.csv'.
    """
    # Sélection des fichiers d'entrée et de sortie en fonction du mode (train/test) :
    if input_file is None:
        input_file = 'data/datasets/x_train.csv' if train_mode else 'data/datasets/x_test.csv'
    if output_file is None:
        output_file = 'y_train.csv' if train_mode else 'y_test.csv'

    # Chargement du fichier d'entrée :
    print(f"Chargement du fichier '{input_file}' :")
    df = get_dataframe(input_file)

    # Extraction des colonnes à prédire 'holed_*' :
    holed_to_predict = [
        column for column in df.columns if 'holed_' in column
    ]
    if not holed_to_predict:
        raise ValueError("Aucune colonne cible 'holed_' trouvée dans le fichier d'entrée.")

    # Extration des colonnes d'entrainement (complètes et non-constantes) :
    potential_predictors = df.columns.difference(holed_to_predict)
    stds = df[potential_predictors].std()
    complete_columns = stds[stds > 1e-9].index.tolist() # On enlève les colonnes constantes car elles sont (quasiment) inutiles pour Ridge et HGBR.

    print(f"\t- Colonnes à prédire (holed)\t = {len(holed_to_predict)}\n\t- Colonnes d'entraînement\t = {len(complete_columns)}")

    # Lancement de l'imputation des données :
    print(f"\nImputation de {len(holed_to_predict)} colonnes :")
    holed_predicted = impute_holed_columns(df, holed_to_predict, complete_columns)
    # Sauvegarde du résultat :
    print(f"\nEnregistrement des prédictions dans '{output_file}'.")
    holed_predicted.to_csv(output_file, index=False)

    # Si on est en mode train (utilise le dataset x_train), on affiche la MAE et le R² associé :
    if train_mode:
        holed_true = get_dataframe('data/datasets/y_train_true.csv').filter(like='holed_')
        holed_nans = get_dataframe('data/datasets/x_train.csv').filter(like='holed_')

        mae, r2 = get_metrics(holed_true, holed_predicted.filter(like='holed_'), holed_nans)
        print(f"\nMétriques [MODE TRAIN] :\n\t- MAE\t = {mae}\n\t- R2\t = {r2}")


if __name__ == "__main__":
    # Détermination des meilleurs hyperparamètres via Optuna :
    best_parameters = find_optimized_parameters() # On devrait normalement définir les hypermaramètres optimaux ici, mais par soucis de temps de calcul, on utilisera ceux déjà trouvés (calculés chez moi) dans les constantes en haut du fichier.

    # Lancement de l'imputation sur le dataset de test (le mode train servant à tester l'efficacité de notre modèle sur 'x_train.csv') :
    run_imputation()

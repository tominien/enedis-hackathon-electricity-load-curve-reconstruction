from typing import Tuple
import csv
import numpy as np
import pandas as pd




def get_dataframe(path: str) -> pd.DataFrame:
    """
    Lit le fichier CSV donné et retourne un DataFrame pandas de ce fichier avec la colonne 'Horodate' en tant qu'index.

    Args:
        path (str): Chemin vers le fichier CSV à lire.
    Raises:
        ValueError: Si la colonne 'Horodate' est absente du CSV.
    Returns:
        pd.DataFrame: DataFrame pandas avec les données du CSV, sans la colonne 'Horodate'.
    """
    df = pd.read_csv(path)

    if 'Horodate' not in df.columns:
        raise ValueError("Le fichier CSV doit contenir une colonne 'Horodate'.")

    df.set_index('Horodate', inplace=True)

    return df


def split_csv_columns(input_csv: str, output_csv: str, start_col: int, end_col: int) -> None:
    """
    Extrait les colonnes d'un CSV entre start_col et end_col (inclus), en conservant la première colonne 'Horodate'.

    Args:
        input_csv (str): Chemin vers le fichier CSV d'entrée.
        output_csv (str): Chemin vers le fichier CSV de sortie.
        start_col (int): Index de la première colonne à extraire (1-based).
        end_col (int): Index de la dernière colonne à extraire (1-based).
    Raises:
        ValueError: Si start_col est supérieur à end_col.
    Returns:
        None
    """
    if start_col > end_col:
        raise ValueError("La colonne de début doit être avant la colonne de fin.")

    with open(input_csv, "r", encoding="utf-8") as infile, open(output_csv, "w", encoding="utf-8") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # Remplissage du CSV de sortie avec la colonne 'Horodate' et les colonnes sélectionnées :
            sliced = [row[0]] + row[start_col - 1:end_col]
            writer.writerow(sliced)


def get_metrics(holed_full: pd.DataFrame, holed_predicted: pd.DataFrame, holed_nans: pd.DataFrame) -> Tuple[float, float]:
    """
    Calcule la MAE et le R² pour un ensemble de colonnes trouées données, uniquement sur les positions contenant des trous (NaNs) à l'origine.
    Ignore les positions où holed_full ou holed_predicted sont NaN.

    Args:
        holed_full (pd.DataFrame): DataFrame avec les vraies valeurs complètes.
        holed_predicted (pd.DataFrame): DataFrame avec les valeurs prédites.
        holed_nans (pd.DataFrame): DataFrame avec les trous (NaNs) d'origine.
    Raises:
        ValueError: Si les shapes des DataFrames ne correspondent pas.
    Returns:
        Tuple[float, float]: MAE et R² calculés sur les positions initialement manquantes.
    """
    # On s'assure que les shapes sont compatibles :
    if holed_full.shape != holed_predicted.shape or holed_full.shape != holed_nans.shape:
        raise ValueError(
            f"Shapes incompatibles :\n\t- holed_full = {holed_full.shape}\n\t- holed_predicted = {holed_predicted.shape}\n\t- holed_nans = {holed_nans.shape}"
        )

    # On masque des trous d'origine :
    mask_missing_values = holed_nans.isna().values # Renvoie un tableau 2D de booléens.

    # On récupère les valeurs réelles et prédites :
    true_holed = holed_full.values
    predicted_holed = holed_predicted.values

    # On liste les positions de chaque valeur prédite et si aucune n'est valide, on retourne (-1.0, 0.0) :
    valid_mask = (
        mask_missing_values
        & np.isfinite(true_holed)
        & np.isfinite(predicted_holed)
    )
    if valid_mask.sum() == 0:
        return -1.0, 0.0

    # On extrait les valeurs valides :
    true_values = true_holed[valid_mask]
    pred_values = predicted_holed[valid_mask]

    # Calcul de la MAE :
    diff = np.abs(true_values - pred_values)
    mae = float(diff.mean())

    # Calcul du R² :
    ss_res = float(np.sum((true_values - pred_values) ** 2))
    ss_tot = float(np.sum((true_values - true_values.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return mae, r2


if __name__ == "__main__":
    REFERENCE_TRAINING_FILE = 'data/datasets/x_train.csv' # Récupérer les colonnes holed_* avec leurs NaNs
    REFERENCE_TESTING_FILE  = 'data/datasets/y_train_true.csv' # Récupérer les colonnes holed_* avec leurs vraies valeurs
    FILE_TO_TEST            = 'y_train.csv' # Récupérer les colonnes holed_* avec les valeurs prédites

    # Load data :
    holed_true_values      = get_dataframe(REFERENCE_TESTING_FILE).filter(like='holed_')
    holed_predicted_values = get_dataframe(FILE_TO_TEST).filter(like='holed_')
    holed_with_nans        = get_dataframe(REFERENCE_TRAINING_FILE).filter(like='holed_')

    # Compute metrics :
    mae, r2 = get_metrics(holed_true_values, holed_predicted_values, holed_with_nans)
    print(f"Métriques :\n\t- MAE\t = {mae}\n\t- R2\t = {r2}")
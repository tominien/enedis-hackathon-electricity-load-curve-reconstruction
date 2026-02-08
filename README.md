# Enedis Linky Load Curve Reconstruction

Code and notebook created by **Tom Le Ber**.

Built an end-to-end pipeline (feature engineering + ridge regression and gradient boosting) to impute missing segments in daily electricity load curves from Enedis Linky smart meter, achieving the best performance in my AI course hackathon.

Challenge project for the Enedis competition hosted on the ENS ChallengeData platform.
Challenge website: [https://challengedata.ens.fr/participants/challenges/160/](https://challengedata.ens.fr/participants/challenges/160/).

---

## General Overview

I strongly recommend visiting the challenge website for more detailed information.

In short, the goal of this project is to reconstruct 24-hour electricity load curves from partially observed data:
- Each file contains a serie of multiple succeding full-day load curve (with 48 half-hour time steps), where:
  - The `x_train.csv` file contain 20000 full columns followed by 1000 partially complete curves, named `holed_1` to `holed_1000`.
  - The `y_train.csv` file contain the last 1000 lines of the `x_train.csv` file completed, with the right data.
  - The `x_test.csv` file contain ~38000 full columns followed by 1000 partially complete curves, named `holed_1` to `holed_1000`.
- The goal of this hackathon is to create a model that would accurately predict the missing power values of the last 1000 lines of `x_test.csv` (like `y_train.csv` from the `x_train.csv` data).
- Performance is evaluated using a regression error on the missing time steps a simple MAE (Mean Absolute Error).

All the code written in the Jupyter Notebook can be found in the `solution.py` and `utils.py` files in this same GitHub repository.

Please note that the files `data/datasets/x_train.csv` and `data/datasets/x_test.csv` could not be included in this repository because they exceed GitHub’s allowed file size limit (100 MB).
You must download them from the ChallengeData page and place them under `data/datasets/`.

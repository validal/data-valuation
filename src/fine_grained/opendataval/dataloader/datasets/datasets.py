"""Default data sets."""
from typing import Union
import os

import numpy as np
import pandas as pd
import sklearn.datasets as ds
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, minmax_scale

from opendataval.dataloader.register import Register, cache


def load_openml(data_id: int, is_classification=True):
    """Load OpenML datasets robustly, handling categorical features.

    - Uses as_frame=True to get DataFrame/Series for safe dtype handling
    - One-hot encodes non-numeric feature columns
    - Converts to numeric numpy arrays (float32 for X, int for y)
    - Standardizes X per column (mean/std) to match existing behavior
    """
    ds = fetch_openml(data_id=data_id, as_frame=True)

    # Extract features and target as pandas objects
    X_df = ds.data.copy()
    y_ser = ds.target

    # Ensure purely numeric features: one-hot encode any non-numeric columns
    non_numeric_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X_df = pd.get_dummies(X_df, columns=non_numeric_cols, drop_first=False)

    # Fill any missing values to avoid NaN propagation during standardization
    X_df = X_df.fillna(0)

    # Label transformation
    if is_classification:
        # Map to integer class codes starting at 0
        y = y_ser.astype("category").cat.codes.to_numpy()
    else:
        y = y_ser.astype(float).to_numpy()
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)

    # Convert features to float32 numpy and standardize
    X = X_df.astype("float32").to_numpy()
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / (X_std + 1e-8)

    return X, y


def load_openml_by_name(name: str, is_classification: bool = True, version: Union[str, int] = "active"):
    """Load OpenML dataset by name with robust categorical handling."""
    ds = fetch_openml(name=name, version=version, as_frame=True)

    X_df = ds.data.copy()
    y_ser = ds.target

    # One-hot non-numeric columns
    non_numeric_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X_df = pd.get_dummies(X_df, columns=non_numeric_cols, drop_first=False)

    X_df = X_df.fillna(0)

    if is_classification:
        y = y_ser.astype("category").cat.codes.to_numpy()
    else:
        y = y_ser.astype(float).to_numpy()
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)

    X = X_df.astype("float32").to_numpy()
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / (X_std + 1e-8)
    return X, y


@Register("gaussian_classifier", one_hot=True)
def gaussian_classifier(n: int = 1000000, input_dim: int = 100):
    """Binary category data set registered as ``"gaussian_classifier"``.

    Artificially generated gaussian noise data set.
    """
    covar = np.random.normal(size=(n, input_dim))

    beta_true = np.random.normal(size=input_dim).reshape(input_dim, 1)
    p_true = np.exp(covar.dot(beta_true)) / (1.0 + np.exp(covar.dot(beta_true)))

    labels = np.random.binomial(n=1, p=p_true).reshape(-1)

    return covar, labels
@Register("gaussian_classifier_high", one_hot=True)
def gaussian_classifier_high(n: int = 1000000, input_dim: int = 100):
    """Binary category data set registered as ``"gaussian_classifier_high"``.

    Artificially generated gaussian noise data set.
    """
    covar = np.random.normal(size=(n, input_dim))

    beta_true = np.random.normal(size=input_dim).reshape(input_dim, 1)
    p_true = np.exp(covar.dot(beta_true)) / (1.0 + np.exp(covar.dot(beta_true)))

    labels = np.random.binomial(n=1, p=p_true).reshape(-1)

    return covar, labels



adult_dataset = Register("adult", one_hot=True, cacheable=True)


@adult_dataset.add_covar_transform(StandardScaler().fit_transform)
def download_adult(cache_dir: str, force_download: bool = False):
    """Binary category data set registered as ``"adult"``. Adult Income data set.

    Implementation from DVRL repository.

    References
    ----------
    .. [1] R. Kohavi, Scaling Up the Accuracy of
        Naive-Bayes Classifiers: a Decision-Tree Hybrid,
        Proceedings of the Second International Conference on Knowledge Discovery
        and Data Mining, 1996
    .. [2] J. Yoon, Arik, Sercan O, and T. Pfister,
        Data Valuation using Reinforcement Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1909.11671.
    """
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
    train_path = cache(uci_url + "/adult.data", cache_dir, "train.csv", force_download)
    test_path = cache(uci_url + "/adult.test", cache_dir, "test.csv", force_download)

    data_train = pd.read_csv(train_path, header=None)
    data_test = pd.read_csv(test_path, skiprows=1, header=None)

    df = pd.concat((data_train, data_test), axis=0)

    # Column names
    df.columns = [
        "Age",
        "WorkClass",
        "fnlwgt",
        "Education",
        "EducationNum",
        "MaritalStatus",
        "Occupation",
        "Relationship",
        "Race",
        "Gender",
        "CapitalGain",
        "CapitalLoss",
        "HoursPerWeek",
        "NativeCountry",
        "Income",
    ]

    # Creates binary labels
    df["Income"] = df["Income"].map(
        {" <=50K": 0, " >50K": 1, " <=50K.": 0, " >50K.": 1}
    )

    # Changes string to float
    df.Age = df.Age.astype(float)
    df.fnlwgt = df.fnlwgt.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.CapitalGain = df.CapitalGain.astype(float)
    df.CapitalLoss = df.CapitalLoss.astype(float)

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=[
            "WorkClass",
            "Education",
            "MaritalStatus",
            "Occupation",
            "Relationship",
            "Race",
            "Gender",
            "NativeCountry",
        ],
    )

    # Sets label name as Y
    df = df.rename(columns={"Income": "Income"})
    df["Income"] = df["Income"].astype(int)

    # Resets index
    df = df.reset_index()
    df = df.drop(columns=["index"])
    return df.drop("Income", axis=1).values, df["Income"].values


# Adult dataset with integer-encoded categorical features (no one-hot on covariates)
adult_int_dataset = Register("adult_int", one_hot=True, cacheable=True)


def _adult_base(cache_dir: str, force_download: bool = False) -> pd.DataFrame:
    """Shared loader that returns the full Adult dataframe with typed columns."""
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
    train_path = cache(uci_url + "/adult.data", cache_dir, "train.csv", force_download)
    test_path = cache(uci_url + "/adult.test", cache_dir, "test.csv", force_download)

    data_train = pd.read_csv(train_path, header=None)
    data_test = pd.read_csv(test_path, skiprows=1, header=None)
    df = pd.concat((data_train, data_test), axis=0)

    df.columns = [
        "Age",
        "WorkClass",
        "fnlwgt",
        "Education",
        "EducationNum",
        "MaritalStatus",
        "Occupation",
        "Relationship",
        "Race",
        "Gender",
        "CapitalGain",
        "CapitalLoss",
        "HoursPerWeek",
        "NativeCountry",
        "Income",
    ]

    # Label to {0,1}
    df["Income"] = df["Income"].map(
        {" <=50K": 0, " >50K": 1, " <=50K.": 0, " >50K.": 1}
    )

    # Cast numerics
    for col in ["Age", "fnlwgt", "EducationNum", "CapitalGain", "CapitalLoss"]:
        df[col] = df[col].astype(float)

    return df.reset_index(drop=True)


@adult_int_dataset.add_covar_transform(lambda X: X)  # no scaling by default
def download_adult_int(cache_dir: str, force_download: bool = False):
    """Adult dataset variant with integer-encoded categorical features.

    - Categorical columns are encoded with category codes (0..K-1) instead of one-hot
    - Numeric columns kept as floats
    - Labels are 0/1 integers; with ExperimentMediator(label_encoding="integer") the
      training labels will remain integers; otherwise they can be converted to one-hot
      by the Experiment setup.
    """
    df = _adult_base(cache_dir, force_download)

    cat_cols = [
        "WorkClass",
        "Education",
        "MaritalStatus",
        "Occupation",
        "Relationship",
        "Race",
        "Gender",
        "NativeCountry",
    ]

    # Integer-encode categorical covariates
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    covar = df.drop("Income", axis=1).astype("float32").values
    labels = df["Income"].astype(int).values
    return covar, labels


# HIGGS dataset (UCI) — large-scale binary classification (up to 11M rows)
# Source: https://archive.ics.uci.edu/ml/datasets/HIGGS
# File:   https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
higgs_dataset = Register("higgs", one_hot=True, cacheable=True)


@higgs_dataset.add_covar_transform(StandardScaler().fit_transform)
def download_higgs(cache_dir: str, force_download: bool = False):
        """Binary classification data set registered as "higgs" (UCI HIGGS).

        Notes
        -----
        - Very large: 11M rows, 28 features. The first column is the label (0/1).
        - To limit memory/time while developing, set environment variable ODV_HIGGS_NROWS
            to an integer (e.g., 1000000 for 1M rows). Default loads 1,000,000 rows.
        """
        base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280"
        gz_path = cache(base_url + "/HIGGS.csv.gz", cache_dir, "HIGGS.csv.gz", force_download)

        # Allow users to limit rows via environment variable
        default_nrows = int(os.environ.get("ODV_HIGGS_NROWS", "1000000"))

        # Read with pandas; gzip is auto-detected by extension
        df = pd.read_csv(gz_path, header=None, nrows=default_nrows)

        # Column 0 is label, remaining 28 columns are features
        labels = df.iloc[:, 0].astype(int).values
        covar = df.iloc[:, 1:].astype("float32").values

        return covar, labels


@Register("iris", one_hot=True)
def download_iris():
    """Categorical data set registered as ``"iris"``."""
    return ds.load_iris(return_X_y=True)


@Register("digits", one_hot=True)
def download_digits():
    """Categorical data set registered as ``"digits"``."""
    return ds.load_digits(return_X_y=True)


@Register("breast_cancer", True).add_covar_transform(minmax_scale)
def download_breast_cancer():
    """Categorical data set registered as ``"breast_cancer"``."""
    return ds.load_breast_cancer(return_X_y=True)

# @Register("breast_cancer", is_classification=True)
# def download_breast_cancer():
#     return ds.load_breast_cancer(return_X_y=True)

# Apply transform after registration
# Register.datasets["breast_cancer"].add_covar_transform(minmax_scale)


@Register("election", one_hot=True, cacheable=True)
def download_election(cache_dir: str, force_download: bool):
    """Categorical data set registered as ``"election"``.

    Presidential election results by MIT Election Data and Science Lab.

    References
    ----------
    .. [1] M. E. Data and S. Lab,
        U.S. President 1976-2020.
        Harvard Dataverse, 2017. doi: 10.7910/DVN/42MVDX.
    """
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    url = "https://dataverse.harvard.edu/api/access/datafile/4299753?gbrecs=false"
    filepath = cache(url, cache_dir, "1976-2020-president.tab", force_download)

    df = pd.read_csv(filepath, delimiter="\t")
    drop_col = [
        "notes",
        "party_detailed",
        "candidate",
        "version",
        "state_po",
        "writein",
        "office",
    ]

    df = df.drop(drop_col, axis=1)
    df = pd.get_dummies(df, columns=["state"])

    covar = df.drop("party_simplified", axis=1).astype("float").values
    labels = df["party_simplified"].astype("category").cat.codes.values

    return covar, labels


# Alternative registration methods, should only be used on ad-hoc basis
Register("gaussian_classifier_high_dim", one_hot=True).from_covar_label_func(
    gaussian_classifier, input_dim=100
)
"""Registers gaussian classifier, but the input_dim is changed."""


# Regression data sets.
@Register("diabetes")
def download_diabetes():
    """Regression data set registered as ``"diabetes"``."""
    return ds.load_diabetes(return_X_y=True)


@Register("linnerud")
def download_linnerud():
    """Regression data set registered as ``"linnerud"``."""
    return ds.load_linnerud(return_X_y=True)


# OpenML Classification Datasets
@Register("2dplanes", one_hot=True)
def download_2dplanes():
    """Categorical data set registered as ``"2dplanes"``."""
    return load_openml(data_id=727)


@Register("electricity", one_hot=True)
def download_electricity():
    """Categorical data set registered as ``"electricity"``."""
    return load_openml(data_id=44080)

@Register("higgs1M", one_hot=True)
def download_higgs1M():
    """Categorical data set registered as ``"higgs1M"``."""
    return load_openml(data_id=42769)
    


@Register("MiniBooNE", one_hot=True)
def download_MiniBooNE():
    """Categorical data set registered as ``"MiniBooNE"``."""
    return load_openml(data_id=43974)


@Register("hepmass_uci", one_hot=True)
def download_hepmass_uci():
    """HEPMASS dataset from local UCI gzip files (1000_train/test.csv.gz).

    Expects under ``data_files/``:
      - ``1000_train.csv.gz``
      - ``1000_test.csv.gz``
    Each with label column '# label' and features f0..f25.
    """
    train_path = os.path.join("data_files", "1000_train.csv.gz")
    test_path = os.path.join("data_files", "1000_test.csv.gz")

    df_train = pd.read_csv(train_path, compression="gzip")
    df_test = pd.read_csv(test_path, compression="gzip")

    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    labels = df["# label"].astype(int).values
    covar = df.drop(columns=["# label"]).astype("float32").values

    return covar, labels


@Register("pol", one_hot=True)
def download_pol():
    """Categorical data set registered as ``"pol"``."""
    return load_openml(data_id=722)


@Register("fried", one_hot=True)
def download_fried():
    """Categorical data set registered as ``"fried"``."""
    return load_openml(data_id=901)


@Register("nomao", one_hot=True)
def download_nomao():
    """Categorical data set registered as ``"nomao"``."""
    return load_openml(data_id=1486)


@Register("creditcard", one_hot=True)
def download_creditcard():
    """Categorical data set registered as ``"creditcard"``."""
    return load_openml(data_id=42477)

@Register("airlines", one_hot=True)
def download_airlines():
    """Categorical data set registered as ``"airlines"``."""
    return load_openml(data_id=45072)



# Additional well-known tabular classification datasets (OpenML)
@Register("bank_marketing", one_hot=True)
def download_bank_marketing():
    """Categorical data set registered as ``"bank_marketing"`` (UCI Bank Marketing)."""
    return load_openml(data_id=1461)


@Register("connect4", one_hot=True)
def download_connect4():
    """Categorical data set registered as ``"connect4"`` (UCI Connect-4)."""
    return load_openml(data_id=40668)


@Register("magic_telescope", one_hot=True)
def download_magic_telescope():
    """Categorical data set registered as ``"magic_telescope"`` (UCI MAGIC Gamma)."""
    return load_openml(data_id=1120)


@Register("phishing_websites", one_hot=True)
def download_phishing_websites():
    """Categorical data set registered as ``"phishing_websites"`` (UCI Phishing Websites)."""
    return load_openml(data_id=4534)


# OpenML Regression Datasets
@Register("wave_energy")
def download_wave_energy():
    """Regression data set registered as ``"wave_energy"``."""
    return load_openml(data_id=44975, is_classification=False)


@Register("lowbwt")
def download_lowbwt():
    """Regression data set registered as ``"lowbwt"``."""
    return load_openml(data_id=1193, is_classification=False)


@Register("mv")
def download_mv():
    """Regression data set registered as ``"mv"``."""
    return load_openml(data_id=344, is_classification=False)


@Register("stock")
def download_stock():
    """Regression data set registered as ``"stock"``."""
    return load_openml(data_id=1200, is_classification=False)


@Register("echoMonths")
def download_echoMonths():
    """Regression data set registered as ``"echoMonths"``."""
    return load_openml(data_id=1199, is_classification=False)


# Discrete-only classification datasets (UCI/OpenML)

@Register("mushroom", one_hot=True)
def download_mushroom():
    """UCI Mushroom — 8,124 rows; Total features: 22 (Categorical: 22, Continuous: 0); Classes: 2.

    Notes
    -----
    All features are nominal (discrete) and will be one-hot encoded.
    """
    return load_openml_by_name(name="mushroom")

@Register("tic_tac_toe", one_hot=True)
def download_tic_tac_toe():
    """UCI Tic-Tac-Toe — 958 rows; Total features: 9 (Categorical: 9, Continuous: 0); Classes: 2.

    Notes
    -----
    Board cells are categorical (x/o/b) and will be one-hot encoded.
    """
    return load_openml_by_name(name="tic-tac-toe")

@Register("car_evaluation", one_hot=True)
def download_car_evaluation():
    """UCI Car Evaluation — 1,728 rows; Total features: 6 (Categorical: 6, Continuous: 0); Classes: 4.

    Notes
    -----
    All attributes are categorical (discrete) and will be one-hot encoded.
    """
    return load_openml_by_name(name="car")

@Register("nursery", one_hot=True)
def download_nursery():
    """UCI Nursery — 12,960 rows; Total features: 8 (Categorical: 8, Continuous: 0); Classes: 5.

    Notes
    -----
    All attributes are categorical (discrete) and will be one-hot encoded.
    """
    return load_openml_by_name(name="nursery")

@Register("kr_vs_kp", one_hot=True)
def download_kr_vs_kp():
    """UCI Chess KR-vs-KP — 3,196 rows; Total features: 36 (Categorical: 36, Continuous: 0); Classes: 2.

    Notes
    -----
    All features are nominal encodings of chess positions; one-hot encoded.
    """
    return load_openml_by_name(name="kr-vs-kp")

@Register("poker_hand", one_hot=True)
def download_poker_hand():
    """UCI Poker Hand — 1,025,010 rows; Total features: 10 (Categorical/Discrete: 10, Continuous: 0); Classes: 10.

    Notes
    -----
    Features are integer-coded (5 suits, 5 ranks). Treated as discrete and one-hot encoded.
    Large dataset; fetching may be slow/memory-heavy.
    """
    # Large dataset; fetching may be slow/memory-heavy.
    return load_openml_by_name(name="poker-hand")

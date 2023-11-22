import pandas as pd
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# Convert fractional year to pandas date format
def fractional_year_to_datetime(year_fraction):
    year = int(year_fraction)
    fraction = year_fraction - year
    delta = pd.Timedelta(fraction * 365.25, unit="d")
    return pd.to_datetime(str(year)) + delta


def load_dataset():
    dataset = {
        "ssd": pd.read_csv(
            "./data/ssd-jcmit.csv", sep=" ", header=None, names=["ds", "y"]
        ),
        "drives": pd.read_csv(
            "./data/drives-jcmit.csv", sep=" ", header=None, names=["ds", "y"]
        ),
        "flash": pd.read_csv(
            "./data/flash-jcmit.csv", sep=" ", header=None, names=["ds", "y"]
        ),
        "memory": pd.read_csv(
            "./data/memory-jcmit.csv", sep=" ", header=None, names=["ds", "y"]
        ),
    }

    for key in dataset:
        dataset[key]["ds"] = dataset[key]["ds"].apply(fractional_year_to_datetime)

    return dataset

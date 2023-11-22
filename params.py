from prophet import Prophet
import optuna
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
from data import load_dataset
from datetime import datetime
import pickle


def test_params(dataset, params, with_floor):
    train, test = dataset

    m = Prophet(**params)
    m.fit(train)

    future = []
    if params["growth"] == "logistic":
        future = test[["ds", "cap"]]
        if with_floor:
            future = test[["ds", "cap", "floor"]]
    else:
        future = test[["ds"]]

    forecast = m.predict(future)

    # Calculate loss
    y_true_part = test["y"].values
    y_pred_part = forecast["yhat"].values

    return y_true_part, y_pred_part


def get_best_params(dataset, n_trials):
    for key in dataset.keys():
        dataset[key]["y"] = np.log(dataset[key]["y"])
        scaler = StandardScaler()
        dataset[key]["y"] = scaler.fit_transform(dataset[key][["y"]])

    def objective(trial):
        params = {
            "growth": trial.suggest_categorical(
                "growth", ["linear", "logistic", "flat"]
            ),
            "yearly_seasonality": trial.suggest_categorical(
                "yearly_seasonality", [True, False]
            ),
            "weekly_seasonality": trial.suggest_categorical(
                "weekly_seasonality", [True, False]
            ),
            "daily_seasonality": trial.suggest_categorical(
                "daily_seasonality", [True, False]
            ),
            "seasonality_mode": trial.suggest_categorical(
                "seasonality_mode", ["additive", "multiplicative"]
            ),
            "seasonality_prior_scale": trial.suggest_loguniform(
                "seasonality_prior_scale", 0.01, 10
            ),
            "changepoint_prior_scale": trial.suggest_loguniform(
                "changepoint_prior_scale", 0.001, 0.5
            ),
        }

        y_true = []
        y_pred = []

        for key in dataset.keys():
            length = len(dataset[key])

            train = dataset[key][: int(length * 0.8)]
            test = dataset[key][int(length * 0.8) :]

            with_floor = trial.suggest_categorical("withFloor", [True, False])
            multiplier = trial.suggest_float("multiplier", 0.8, 2137)

            if params["growth"] == "logistic":
                cap = train["y"].max() * multiplier  # No cap 🧢
                dataset[key]["cap"] = cap
                if with_floor:
                    floor = train["y"].min() * multiplier
                    dataset[key]["floor"] = floor
                elif "floor" in dataset[key]:
                    del dataset[key]["floor"]
            else:
                if "cap" in dataset[key]:
                    del dataset[key]["cap"]
                if "floor" in dataset[key]:
                    del dataset[key]["floor"]

            train = dataset[key][: int(length * 0.8)]
            test = dataset[key][int(length * 0.8) :]

            y_true_part, y_pred_part = test_params((train, test), params, with_floor)

            y_true.extend(y_true_part)
            y_pred.extend(y_pred_part)

        loss = mean_squared_error(y_true, y_pred)

        return loss

    storage = optuna.storages.RDBStorage(
        url="sqlite:///db.sqlite3", engine_kwargs={"connect_args": {"timeout": 420}}
    )
    study = optuna.create_study(
        storage=storage,
        direction="minimize",
        study_name=f"{datetime.now()}",
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
    return study.best_trial


if __name__ == "__main__":
    dataset = load_dataset()
    params = get_best_params(dataset, 5000).params

    with open("params.dict", "wb") as params_dict_file:
        pickle.dump(params, params_dict_file)

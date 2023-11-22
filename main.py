from prophet import Prophet
from multiprocessing import Pool
from data import load_dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


def get_prediction(dataset, key, params):
    dataset[key]["y"] = np.log(dataset[key]["y"])
    scaler = StandardScaler()
    dataset[key]["y"] = scaler.fit_transform(dataset[key][["y"]])

    cap = dataset[key]["y"].max() * params["multiplier"]
    floor = dataset[key]["y"].min() * params["multiplier"]
    del params["multiplier"]

    dataset[key]["cap"] = cap  # No cap 🧢
    if (params["withFloor"]):
        dataset[key]["floor"] = floor
    del params["withFloor"]

    m = Prophet(**params)
    m.fit(dataset[key])

    future = m.make_future_dataframe(periods=5000)
    future["cap"] = cap
    if ("floor" in dataset[key]):
        future["floor"] = floor

    forecast = m.predict(future)

    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)

    fig1.savefig(f"./images/fig1_{key}.png")
    fig2.savefig(f"./images/fig2_{key}.png")

    return (m, forecast)


if __name__ == "__main__":
    dataset = load_dataset()
    params = pickle.load(open("./params.dict", "rb"))

    with Pool(processes=4) as pool:
        results = pool.starmap(
            get_prediction, [(dataset, key, params) for key in dataset.keys()]
        )

        for m, forecast in results:
            fig1 = m.plot(forecast)
            fig2 = m.plot_components(forecast)

        pool.terminate()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)


TRANSPORT_FACTORS = {
    "Car": 0.21,
    "Bike": 0.08,
    "Bus": 0.05,
    "Train": 0.04,
    "Walking": 0.00,
}

DIET_FACTORS = {
    "Vegan": 2.0,
    "Vegetarian": 2.5,
    "Mixed": 4.0,
    "Non-Vegetarian": 5.0,
}

ELECTRICITY_FACTOR = 0.82
WASTE_FACTOR = 0.50


@dataclass(frozen=True)
class EmissionResult:
    transport: float
    electricity: float
    food: float
    waste: float
    total: float
    highest_category: str
    recommendation: str


def calculate_emissions(
    transport_type: str,
    distance_km: float,
    electricity_units: float,
    diet_type: str,
    waste_kg: float,
) -> EmissionResult:
    transport = round(distance_km * TRANSPORT_FACTORS[transport_type], 2)
    electricity = round(electricity_units * ELECTRICITY_FACTOR, 2)
    food = round(DIET_FACTORS[diet_type], 2)
    waste = round(waste_kg * WASTE_FACTOR, 2)
    total = round(transport + electricity + food + waste, 2)

    categories = {
        "Transport": transport,
        "Electricity": electricity,
        "Food": food,
        "Waste": waste,
    }
    highest_category = max(categories, key=categories.get)

    recommendations = {
        "Transport": "Use public transport, carpool, cycle, or walk for shorter trips to reduce travel emissions.",
        "Electricity": "Switch off unused appliances, use efficient lighting, and monitor high-consumption devices.",
        "Food": "Choose more plant-based meals and reduce high-emission food choices where possible.",
        "Waste": "Reduce, reuse, recycle, and separate organic waste to lower disposal-related emissions.",
    }

    return EmissionResult(
        transport=transport,
        electricity=electricity,
        food=food,
        waste=waste,
        total=total,
        highest_category=highest_category,
        recommendation=recommendations[highest_category],
    )


def generate_dataset() -> pd.DataFrame:
    np.random.seed(42)
    user_count = 100

    data = {
        "user_id": range(1, user_count + 1),
        "transport_type": np.random.choice(list(TRANSPORT_FACTORS), user_count),
        "distance_km": np.random.randint(5, 151, user_count),
        "electricity_units": np.random.randint(50, 401, user_count),
        "diet_type": np.random.choice(list(DIET_FACTORS), user_count),
        "waste_kg": np.round(np.random.uniform(1, 25, user_count), 2),
    }

    df = pd.DataFrame(data)
    df["transport_emission"] = df.apply(
        lambda row: round(row["distance_km"] * TRANSPORT_FACTORS[row["transport_type"]], 2),
        axis=1,
    )
    df["electricity_emission"] = (df["electricity_units"] * ELECTRICITY_FACTOR).round(2)
    df["food_emission"] = df["diet_type"].map(DIET_FACTORS).round(2)
    df["waste_emission"] = (df["waste_kg"] * WASTE_FACTOR).round(2)
    df["total_emission_kg"] = (
        df["transport_emission"]
        + df["electricity_emission"]
        + df["food_emission"]
        + df["waste_emission"]
    ).round(2)
    return df


DATASET = generate_dataset()


def series_to_bars(series: pd.Series) -> list[dict[str, Any]]:
    max_value = float(series.max()) if len(series) else 1.0
    return [
        {
            "label": str(label),
            "value": round(float(value), 2),
            "percent": round((float(value) / max_value) * 100, 2) if max_value else 0,
        }
        for label, value in series.items()
    ]


def build_eda_context() -> dict[str, Any]:
    category_totals = pd.Series(
        {
            "Transport": DATASET["transport_emission"].sum(),
            "Electricity": DATASET["electricity_emission"].sum(),
            "Food": DATASET["food_emission"].sum(),
            "Waste": DATASET["waste_emission"].sum(),
        }
    ).round(2)

    transport_avg = (
        DATASET.groupby("transport_type")["transport_emission"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )
    diet_avg = (
        DATASET.groupby("diet_type")["food_emission"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )
    top_emitters = DATASET.nlargest(5, "total_emission_kg")[
        [
            "user_id",
            "transport_type",
            "diet_type",
            "transport_emission",
            "electricity_emission",
            "food_emission",
            "waste_emission",
            "total_emission_kg",
        ]
    ].to_dict("records")

    bins = pd.cut(DATASET["total_emission_kg"], bins=6)
    distribution = DATASET.groupby(bins, observed=False).size()
    distribution.index = [f"{int(interval.left)}-{int(interval.right)} kg" for interval in distribution.index]

    highest_category = category_totals.idxmax()
    lowest_category = category_totals.idxmin()

    return {
        "dataset_size": len(DATASET),
        "average_total": round(float(DATASET["total_emission_kg"].mean()), 2),
        "maximum_total": round(float(DATASET["total_emission_kg"].max()), 2),
        "minimum_total": round(float(DATASET["total_emission_kg"].min()), 2),
        "highest_category": highest_category,
        "lowest_category": lowest_category,
        "category_totals": series_to_bars(category_totals),
        "transport_avg": series_to_bars(transport_avg),
        "diet_avg": series_to_bars(diet_avg),
        "distribution": series_to_bars(distribution),
        "top_emitters": top_emitters,
    }


@app.route("/")
def home():
    eda = build_eda_context()
    return render_template("home.html", eda=eda)


@app.route("/estimator", methods=["GET", "POST"])
def estimator():
    result = None
    form_data = {
        "transport_type": "Car",
        "distance_km": 25,
        "electricity_units": 120,
        "diet_type": "Mixed",
        "waste_kg": 5,
    }

    if request.method == "POST":
        form_data = {
            "transport_type": request.form.get("transport_type", "Car"),
            "distance_km": float(request.form.get("distance_km", 0) or 0),
            "electricity_units": float(request.form.get("electricity_units", 0) or 0),
            "diet_type": request.form.get("diet_type", "Mixed"),
            "waste_kg": float(request.form.get("waste_kg", 0) or 0),
        }
        result = calculate_emissions(**form_data)

    return render_template(
        "estimator.html",
        transport_types=list(TRANSPORT_FACTORS),
        diet_types=list(DIET_FACTORS),
        form_data=form_data,
        result=result,
    )


@app.route("/eda")
def eda():
    return render_template("eda.html", eda=build_eda_context())


if __name__ == "__main__":
    app.run(debug=True)

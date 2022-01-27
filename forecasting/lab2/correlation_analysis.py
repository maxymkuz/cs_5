
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression

import altair as alt

alt.data_transformers.disable_max_rows()
alt.renderers.enable('mimetype');

def create_lags_dataset(x, y, n_lags):
    df = pd.DataFrame({"lag0": x})

    for i in range(1, n_lags + 1):
        df[f"lag{i}"] = x.shift(i)

    return df.iloc[n_lags:], y.iloc[n_lags:]

def get_random_forest_importances(x, y):
    model = RandomForestRegressor()
    model.fit(x, y)
    return model.feature_importances_

def get_pacf(x, y):
    columns = x.columns
    pacf = [x[columns[0]].corr(y)]
    
    for i in range(1, len(columns)):
        model = LinearRegression()
        cur_x = x[columns[:i]]
        model.fit(cur_x, y)
        residuals = y - model.predict(cur_x)
        pacf.append(x[columns[i]].corr(residuals))
    
    return np.array(pacf)

def get_feature_importance_df(independent_columns, use_zero_lag_list, target_column, n_lags, feature_importance_fn):
    lags = list(range(n_lags + 1))
    feature_importances_dict = {"lag": lags}
    
    for column, use_zero_lag in zip(independent_columns, use_zero_lag_list):
        x, y = create_lags_dataset(column, target_column, n_lags)
        if not use_zero_lag:
            x.drop("lag0", axis=1, inplace=True)
        
        feature_importances = feature_importance_fn(x, y)
        if not use_zero_lag:
            feature_importances = np.concatenate((np.array([np.NaN]), feature_importances))
        
        feature_importances_dict[column.name] = feature_importances
    
    return pd.DataFrame(feature_importances_dict)

def plot_correlations(main):
    melted_df = pd.melt(main, ["dt", "Johor_Production"])
    independent_columns = ["Johor_Rainfall", 'Johor_Area_rpa', 'Johor_Area_npa', 'Johor_Area_ma_new']
    corr = main.corr()["Johor_Production"]
    mutual_info = mutual_info_regression(main[independent_columns], main["Johor_Production"])

    scatter_plot = alt.Chart(melted_df).mark_point().encode(
        x=alt.X("value:Q", scale=alt.Scale(zero=False)),
        y=alt.Y("Johor_Production:Q", scale=alt.Scale(zero=False))
    )

    return alt.ConcatChart(
        concat=[
        scatter_plot.transform_filter(alt.FieldEqualPredicate(field="variable", equal=name)) \
                    .properties(title=f"column: {name}; correlation: {corr[name]}; mutual information: {mutual_info[i]}", height=300, width=500)
        for i, name in enumerate(independent_columns)
        ],
        columns=2
    ).resolve_axis(
        x="independent",
        y="independent"
    ).resolve_scale(
        x="independent",
        y="independent"
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=11
    ).configure_title(fontSize=13)

def print_corr_anal(main, n_lags=100):
    columns = ["Johor_Rainfall", "Johor_Production", 'Johor_Area_rpa', 'Johor_Area_npa', 'Johor_Area_ma_new']
    use_zero_lag_list = [True, True, False]
    independent_columns = [main[column] for column in columns]
    target_column = main["Johor_Production"]

    random_forest_importances = get_feature_importance_df(independent_columns, use_zero_lag_list, target_column, n_lags, get_random_forest_importances)
    random_forest_importances.head(3)

    chart = alt.Chart(pd.melt(random_forest_importances, "lag")).mark_line(point=True).encode(
    x=alt.X("lag:Q", scale=alt.Scale(zero=False)),
    y=alt.Y("value:Q", scale=alt.Scale(zero=False))
    )

    return alt.ConcatChart(
        concat=[
        chart.transform_filter(alt.FieldEqualPredicate(field="variable", equal=name)) \
                    .properties(title=name, height=300, width=500)
        for name in columns
        ],
        columns=2
    ).resolve_axis(
        x="independent",
        y="independent"
    ).resolve_scale(
        x="independent", 
        y="independent"
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=11
    ).configure_title(fontSize=13)

def part_autocorrelation(main):
    columns = ["Johor_Rainfall", "Johor_Production", 'Johor_Area_rpa', 'Johor_Area_npa', 'Johor_Area_ma_new']
    use_zero_lag_list = [True, True, False]
    independent_columns = [main[column] for column in columns]
    target_column = main["Johor_Production"]

    n_lags = 100
    pacf_importances = get_feature_importance_df(independent_columns, use_zero_lag_list, target_column, n_lags, get_pacf)
    pacf_importances.head(3)
    chart = alt.Chart(pd.melt(pacf_importances, "lag")).mark_line(point=True).encode(
        x=alt.X("lag:Q", scale=alt.Scale(zero=False)),
        y=alt.Y("value:Q", scale=alt.Scale(zero=False))
    )

    return alt.ConcatChart(
        concat=[
        chart.transform_filter(alt.FieldEqualPredicate(field="variable", equal=name)) \
                    .properties(title=name, height=300, width=500)
        for name in columns
        ],
        columns=2
    ).resolve_axis(
        x="independent",
        y="independent"
    ).resolve_scale(
        x="shared", 
        y="shared"
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=11
    ).configure_title(fontSize=13)

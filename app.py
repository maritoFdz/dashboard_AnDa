import copy
import io
import base64

import pandas as pd
import country_converter as coco
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dash import Dash, dcc, html, Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

sns.set_theme(style="whitegrid")

# Carga del dataset

df = pd.read_csv("World_Happiness_limpio.csv", encoding="cp1252")
converter = coco.CountryConverter()
df["Continent"] = converter.convert(names=df["Country name"], to="continent", not_found=None)
df_america = df[df["Continent"] == "America"].copy()
converter = coco.CountryConverter()
df_america["Region"] = converter.convert(names=df_america["Country name"], to="UNregion", not_found=None)

# este es como el del notebook que s e llama df normal
df_mundial = df.dropna(subset=["Continent"]).copy()

num_cols = [c for c in df.select_dtypes(include="number").columns if c != "year"]
continents = sorted(df_mundial["Continent"].unique())
regiosen = sorted(df_america["Region"].dropna().unique())
var_options = [{"label": c, "value": c} for c in num_cols]

# Entrenado del modelo
label = LabelEncoder()
df_encoded = df.dropna().copy()
df_encoded['Continent'] = label.fit_transform(df_encoded['Continent'])
encoder = copy.deepcopy(label)
df_encoded['Country name'] = label.fit_transform(df_encoded['Country name'])

X_df = df_encoded.drop(columns=['Life Ladder', 'Country name', 'year'])
train_cols = X_df.columns.tolist()  # nombres de columnas usadas para entrenar, en el mismo orden
X = X_df.values
Y = df_encoded['Life Ladder'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=11)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

randomForestRegressor = RandomForestRegressor(n_jobs=2, random_state=0)
randomForestRegressor.fit(X_train, Y_train)

input_cols = [c for c in train_cols if c != "Continent"]

# dash necesita imagenes pero usar seaborn es super practcio asi que esto convierte lo de seaborn en fotos para dash
def fig_to_src(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


tam_fig = (7, 4)          # tamaño compacto para las gráficas normales
tam_corr = (6, 5)     # tamaño para la matriz de correlación (más cuadrada)


# Dashboard
app = Dash(__name__)
app.title = "World Happiness Report - Dashboard - Proyecto Final Analisis de Datos y Toma de Decisiones"
server = app.server  # necesario para desplegar en Render/Heroku

app.layout = html.Div([
    html.H1("World Happiness Report - Dashboard"),

    # Pestanas
    dcc.Tabs([
        # Por Continentetes
        dcc.Tab(label="Mundial", children=[
            html.H3("Distribución de una variable"),
            html.Label("Variable"),
            dcc.Dropdown(id="mun-dist-var", options=var_options, value="Life Ladder", clearable=False),
            html.Label("Tipo de gráfico"),
            dcc.RadioItems(
                id="mun-dist-tipo",
                options=[{"label": "Densidad", "value": "densidad"}, {"label": "Violín", "value": "violin"}],
                value="densidad", inline=True,
            ),
            html.Img(id="mun-dist-graph"),

            html.H3("Matriz de correlación"),
            html.Label("Continente"),
            dcc.Dropdown(
                id="mun-corr-continente",
                options=[{"label": "Todos", "value": "Todos"}] + [{"label": c, "value": c} for c in continents],
                value="Todos", clearable=False,
            ),
            html.Img(id="mun-corr-graph"),

            html.H3("Evolución temporal de Life Ladder"),
            html.Label("Continentes a mostrar"),
            dcc.Checklist(
                id="mun-evo-checklist",
                options=[{"label": c, "value": c} for c in continents],
                value=continents, inline=True,
            ),
            html.Img(id="mun-evo-graph"),

            html.H3("Relación entre dos variables"),
            html.Label("Variable X"),
            dcc.Dropdown(id="mun-scatter-x", options=var_options, value="Log GDP per capita", clearable=False),
            html.Label("Variable Y"),
            dcc.Dropdown(id="mun-scatter-y", options=var_options, value="Life Ladder", clearable=False),
            html.Img(id="mun-scatter-graph"),
        ]),

        # America
        dcc.Tab(label="América", children=[
            html.H3("Distribución de una variable"),
            html.Label("Variable"),
            dcc.Dropdown(id="am-dist-var", options=var_options, value="Life Ladder", clearable=False),
            html.Label("Tipo de gráfico"),
            dcc.RadioItems(
                id="am-dist-tipo",
                options=[{"label": "Densidad", "value": "densidad"}, {"label": "Violín", "value": "violin"}],
                value="densidad", inline=True,
            ),
            html.Img(id="am-dist-graph"),

            html.H3("Matriz de correlación"),
            html.Label("Región"),
            dcc.Dropdown(
                id="am-corr-region",
                options=[{"label": "Todas", "value": "Todos"}] + [{"label": r, "value": r} for r in regiosen],
                value="Todos", clearable=False,
            ),
            html.Img(id="am-corr-graph"),

            html.H3("Evolución temporal de Life Ladder"),
            html.Label("Regiones a mostrar"),
            dcc.Checklist(
                id="am-evo-checklist",
                options=[{"label": r, "value": r} for r in regiosen],
                value=regiosen, inline=True,
            ),
            html.Img(id="am-evo-graph"),

            html.H3("Relación entre dos variables"),
            html.Label("Variable X"),
            dcc.Dropdown(id="am-scatter-x", options=var_options, value="Social support", clearable=False),
            html.Label("Variable Y"),
            dcc.Dropdown(id="am-scatter-y", options=var_options, value="Life Ladder", clearable=False),
            html.Img(id="am-scatter-graph"),
        ]),

        # Mapa
        dcc.Tab(label="Mapa", children=[
            html.H3("Mapa mundial interactivo"),
            html.P("Pendiente."),
        ]),

        # Modelo de ML
        dcc.Tab(label="Predicción", children=[
            html.H3("Predicción de Life Ladder"),
            html.P("Ingresa los valores para estimar el nivel de felicidad predicho por el modelo."),

            html.Label("Continente"),
            dcc.Dropdown(id="pred-continente", options=[{"label": c, "value": c} for c in continents],
                         value=continents[0], clearable=False),

            *[
                html.Div([
                    html.Label(col),
                    dcc.Input(id=f"pred-{i}", type="number", value=round(float(df[col].median()), 2), step=0.01),
                ])
                for i, col in enumerate(input_cols)
            ],

            html.Button("Predecir", id="pred-boton", n_clicks=0),
            html.Div(id="pred-resultado"),
        ]),
    ]),
])

# Callbacks

# Mundial
@app.callback(Output("mun-dist-graph", "src"), Input("mun-dist-var", "value"), Input("mun-dist-tipo", "value"))
def actualizar_dist_mundial(variable, tipo):
    fig, ax = plt.subplots(figsize=tam_fig)
    if tipo == "densidad":
        sns.kdeplot(data=df_mundial, x=variable, hue="Continent", palette="deep", ax=ax)
    else:
        sns.violinplot(data=df_mundial, x="Continent", y=variable, hue="Continent", palette="deep", ax=ax)
    ax.set_title(f"Distribución de {variable} por continente")
    return fig_to_src(fig)


@app.callback(Output("mun-corr-graph", "src"), Input("mun-corr-continente", "value"))
def actualizar_corr_mundial(continente):
    data = df_mundial if continente == "Todos" else df_mundial[df_mundial["Continent"] == continente]
    corr = data[num_cols].corr()
    fig, ax = plt.subplots(figsize=tam_corr)
    sns.heatmap(corr, cmap="RdBu", annot=True, fmt=".2f", cbar=False, ax=ax)
    ax.set_title(f"Correlación entre variables ({continente})")
    return fig_to_src(fig)


@app.callback(Output("mun-evo-graph", "src"), Input("mun-evo-checklist", "value"))
def actualizar_evo_mundial(continentes_sel):
    data = df_mundial[df_mundial["Continent"].isin(continentes_sel)]
    fig, ax = plt.subplots(figsize=tam_fig)
    sns.lineplot(data=data, x="year", y="Life Ladder", hue="Continent", palette="deep", marker="o", ax=ax)
    ax.set_title("Evolución de Life Ladder promedio por continente")
    return fig_to_src(fig)


@app.callback(Output("mun-scatter-graph", "src"), Input("mun-scatter-x", "value"), Input("mun-scatter-y", "value"))
def actualizar_scatter_mundial(var_x, var_y):
    fig, ax = plt.subplots(figsize=tam_fig)
    sns.scatterplot(data=df_mundial, x=var_x, y=var_y, hue="Continent", palette="deep", ax=ax)
    ax.set_title(f"{var_y} vs {var_x}")
    return fig_to_src(fig)


# America
@app.callback(Output("am-dist-graph", "src"), Input("am-dist-var", "value"), Input("am-dist-tipo", "value"))
def actualizar_dist_america(variable, tipo):
    fig, ax = plt.subplots(figsize=tam_fig)
    if tipo == "densidad":
        sns.kdeplot(data=df_america, x=variable, hue="Region", palette="Set2", ax=ax)
    else:
        sns.violinplot(data=df_america, x="Region", y=variable, hue="Region", palette="Set2", ax=ax)
        ax.tick_params(axis="x", rotation=20)
    ax.set_title(f"Distribución de {variable} por región de América")
    return fig_to_src(fig)


@app.callback(Output("am-corr-graph", "src"), Input("am-corr-region", "value"))
def actualizar_corr_america(region):
    data = df_america if region == "Todos" else df_america[df_america["Region"] == region]
    corr = data[num_cols].corr()
    fig, ax = plt.subplots(figsize=tam_corr)
    sns.heatmap(corr, cmap=sns.cubehelix_palette(hue=1, as_cmap=True), annot=True, fmt=".2f", cbar=False, ax=ax)
    ax.set_title(f"Correlación entre variables ({region})")
    return fig_to_src(fig)


@app.callback(Output("am-evo-graph", "src"), Input("am-evo-checklist", "value"))
def actualizar_evo_america(regiones_sel):
    data = df_america[df_america["Region"].isin(regiones_sel)]
    fig, ax = plt.subplots(figsize=tam_fig)
    sns.lineplot(data=data, x="year", y="Life Ladder", hue="Region", palette="Set2", marker="o", ax=ax)
    ax.set_title("Evolución de Life Ladder promedio por región de América")
    return fig_to_src(fig)


@app.callback(Output("am-scatter-graph", "src"), Input("am-scatter-x", "value"), Input("am-scatter-y", "value"))
def actualizar_scatter_america(var_x, var_y):
    fig, ax = plt.subplots(figsize=tam_fig)
    sns.scatterplot(data=df_america, x=var_x, y=var_y, hue="Region", palette="Set2", ax=ax)
    ax.set_title(f"{var_y} vs {var_x}")
    return fig_to_src(fig)


# Prediccion
@app.callback(
    Output("pred-resultado", "children"),
    Input("pred-boton", "n_clicks"),
    State("pred-continente", "value"),
    *[State(f"pred-{i}", "value") for i in range(len(input_cols))],
    prevent_initial_call=True,
)
def predecir(n_clicks, continente, *valores):
    if any(v is None for v in valores):
        return "Por favor completa todos los campos."

    continente_cod = encoder.transform([continente])[0]
    fila = {"Continent": continente_cod}
    fila.update(dict(zip(input_cols, valores)))

    entrada = pd.DataFrame([fila])[train_cols]
    entrada_esc = scaler.transform(entrada.values)
    prediccion = randomForestRegressor.predict(entrada_esc)[0]

    return f"Nivel de felicidad estimado: {prediccion:.2f}"

if __name__ == "__main__":
    app.run(debug=True)
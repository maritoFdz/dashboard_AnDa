import pandas as pd
import country_converter as coco
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Carga Dataset
df = pd.read_csv("World_Happiness_limpio.csv", encoding="cp1252")

converter = coco.CountryConverter()
df["Continent"] = converter.convert(names=df["Country name"], to="continent", not_found=None)

df["Region"] = None
mask_america = df["Continent"] == "America"
df.loc[mask_america, "Region"] = converter.convert(
    names=df.loc[mask_america, "Country name"].tolist(), to="UNregion", not_found=None
)

df_mundial = df.dropna(subset=["Continent"]).copy()
df_america = df[df["Continent"] == "America"].dropna(subset=["Region"]).copy()

NUMERIC_COLS = [c for c in df.select_dtypes(include="number").columns if c != "year"]
CONTINENTES = sorted(df_mundial["Continent"].unique())
REGIONES = sorted(df_america["Region"].unique())
var_options = [{"label": c, "value": c} for c in NUMERIC_COLS]

# Entreno Modelo

FEATURE_COLS = [
    "Continent",
    "Log GDP per capita",
    "Social support",
    "Healthy life expectancy at birth",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption",
    "Positive affect",
    "Negative affect",
]

continent_encoder = LabelEncoder()
df_encoded = df_mundial.dropna(subset=FEATURE_COLS + ["Life Ladder"]).copy()
df_encoded["Continent"] = continent_encoder.fit_transform(df_encoded["Continent"])

X = df_encoded[FEATURE_COLS].values
Y = df_encoded["Life Ladder"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=11)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo = RandomForestRegressor(n_jobs=2, random_state=0)
modelo.fit(X_train, Y_train)

print(f"R2 train: {modelo.score(X_train, Y_train):.3f}")
print(f"R2 test:  {modelo.score(X_test, Y_test):.3f}")

FEATURE_INPUT_COLS = [c for c in FEATURE_COLS if c != "Continent"]


# Dashboard

app = Dash(__name__)
app.title = "World Happiness Report - Dashboard - PRoyecto Final Analisis de Datos y Toma de Decisiones"
server = app.server  # necesario para desplegar en Render/Heroku

app.layout = html.Div([
    html.H1("World Happiness Report - Dashboard"),

    dcc.Tabs([
        # Pestana mundial
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
            dcc.Graph(id="mun-dist-graph"),

            html.H3("Matriz de correlación"),
            html.Label("Continente"),
            dcc.Dropdown(
                id="mun-corr-continente",
                options=[{"label": "Todos", "value": "Todos"}] + [{"label": c, "value": c} for c in CONTINENTES],
                value="Todos", clearable=False,
            ),
            dcc.Graph(id="mun-corr-graph"),

            html.H3("Evolución temporal de Life Ladder"),
            html.Label("Continentes a mostrar"),
            dcc.Checklist(
                id="mun-evo-checklist",
                options=[{"label": c, "value": c} for c in CONTINENTES],
                value=CONTINENTES, inline=True,
            ),
            dcc.Graph(id="mun-evo-graph"),

            html.H3("Relación entre dos variables"),
            html.Label("Variable X"),
            dcc.Dropdown(id="mun-scatter-x", options=var_options, value="Log GDP per capita", clearable=False),
            html.Label("Variable Y"),
            dcc.Dropdown(id="mun-scatter-y", options=var_options, value="Life Ladder", clearable=False),
            dcc.Graph(id="mun-scatter-graph"),
        ]),

        # Pestana america
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
            dcc.Graph(id="am-dist-graph"),

            html.H3("Matriz de correlación"),
            html.Label("Región"),
            dcc.Dropdown(
                id="am-corr-region",
                options=[{"label": "Todas", "value": "Todos"}] + [{"label": r, "value": r} for r in REGIONES],
                value="Todos", clearable=False,
            ),
            dcc.Graph(id="am-corr-graph"),

            html.H3("Evolución temporal de Life Ladder"),
            html.Label("Regiones a mostrar"),
            dcc.Checklist(
                id="am-evo-checklist",
                options=[{"label": r, "value": r} for r in REGIONES],
                value=REGIONES, inline=True,
            ),
            dcc.Graph(id="am-evo-graph"),

            html.H3("Relación entre dos variables"),
            html.Label("Variable X"),
            dcc.Dropdown(id="am-scatter-x", options=var_options, value="Social support", clearable=False),
            html.Label("Variable Y"),
            dcc.Dropdown(id="am-scatter-y", options=var_options, value="Life Ladder", clearable=False),
            dcc.Graph(id="am-scatter-graph"),
        ]),

        # Pestana mapa
        dcc.Tab(label="Mapa", children=[
            html.H3("Mapa mundial interactivo"),
            html.Label("Variable"),
            dcc.Dropdown(id="mapa-var", options=var_options, value="Life Ladder", clearable=False),
            html.Label("Año"),
            dcc.Slider(
                id="mapa-year",
                min=int(df["year"].min()), max=int(df["year"].max()), step=1,
                value=int(df["year"].max()),
                marks={int(y): str(int(y)) for y in sorted(df["year"].unique())},
            ),
            dcc.Graph(id="mapa-graph"),
        ]),

        # Pestana prediccion
        dcc.Tab(label="Predicción", children=[
            html.H3("Predicción de Life Ladder"),
            html.P("Ingresa los valores para estimar el nivel de felicidad predicho por el modelo."),

            html.Label("Continente"),
            dcc.Dropdown(id="pred-continente", options=[{"label": c, "value": c} for c in CONTINENTES],
                         value=CONTINENTES[0], clearable=False),

            *[
                html.Div([
                    html.Label(col),
                    dcc.Input(id=f"pred-{i}", type="number", value=round(float(df[col].median()), 2), step=0.01),
                ])
                for i, col in enumerate(FEATURE_INPUT_COLS)
            ],

            html.Button("Predecir", id="pred-boton", n_clicks=0),
            html.Div(id="pred-resultado"),
        ]),
    ]),
])





# Mundial

@app.callback(Output("mun-dist-graph", "figure"), Input("mun-dist-var", "value"), Input("mun-dist-tipo", "value"))
def actualizar_dist_mundial(variable, tipo):
    if tipo == "densidad":
        fig = px.histogram(df_mundial, x=variable, color="Continent", histnorm="probability density",
                            opacity=0.5, barmode="overlay", nbins=40)
    else:
        fig = px.violin(df_mundial, x="Continent", y=variable, color="Continent", box=True)
    fig.update_layout(title=f"Distribución de {variable} por continente")
    return fig


@app.callback(Output("mun-corr-graph", "figure"), Input("mun-corr-continente", "value"))
def actualizar_corr_mundial(continente):
    data = df_mundial if continente == "Todos" else df_mundial[df_mundial["Continent"] == continente]
    corr = data[NUMERIC_COLS].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(title=f"Correlación entre variables ({continente})")
    return fig


@app.callback(Output("mun-evo-graph", "figure"), Input("mun-evo-checklist", "value"))
def actualizar_evo_mundial(continentes_sel):
    data = df_mundial[df_mundial["Continent"].isin(continentes_sel)]
    resumen = data.groupby(["year", "Continent"], as_index=False)["Life Ladder"].mean()
    fig = px.line(resumen, x="year", y="Life Ladder", color="Continent", markers=True)
    fig.update_layout(title="Evolución de Life Ladder promedio por continente")
    return fig


@app.callback(Output("mun-scatter-graph", "figure"), Input("mun-scatter-x", "value"), Input("mun-scatter-y", "value"))
def actualizar_scatter_mundial(var_x, var_y):
    fig = px.scatter(df_mundial, x=var_x, y=var_y, color="Continent", opacity=0.7)
    fig.update_layout(title=f"{var_y} vs {var_x}")
    return fig


# America

@app.callback(Output("am-dist-graph", "figure"), Input("am-dist-var", "value"), Input("am-dist-tipo", "value"))
def actualizar_dist_america(variable, tipo):
    if tipo == "densidad":
        fig = px.histogram(df_america, x=variable, color="Region", histnorm="probability density",
                            opacity=0.5, barmode="overlay", nbins=40)
    else:
        fig = px.violin(df_america, x="Region", y=variable, color="Region", box=True)
    fig.update_layout(title=f"Distribución de {variable} por región de América")
    return fig


@app.callback(Output("am-corr-graph", "figure"), Input("am-corr-region", "value"))
def actualizar_corr_america(region):
    data = df_america if region == "Todos" else df_america[df_america["Region"] == region]
    corr = data[NUMERIC_COLS].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(title=f"Correlación entre variables ({region})")
    return fig


@app.callback(Output("am-evo-graph", "figure"), Input("am-evo-checklist", "value"))
def actualizar_evo_america(regiones_sel):
    data = df_america[df_america["Region"].isin(regiones_sel)]
    resumen = data.groupby(["year", "Region"], as_index=False)["Life Ladder"].mean()
    fig = px.line(resumen, x="year", y="Life Ladder", color="Region", markers=True)
    fig.update_layout(title="Evolución de Life Ladder promedio por región de América")
    return fig


@app.callback(Output("am-scatter-graph", "figure"), Input("am-scatter-x", "value"), Input("am-scatter-y", "value"))
def actualizar_scatter_america(var_x, var_y):
    fig = px.scatter(df_america, x=var_x, y=var_y, color="Region", opacity=0.7)
    fig.update_layout(title=f"{var_y} vs {var_x}")
    return fig


# Mapa

@app.callback(Output("mapa-graph", "figure"), Input("mapa-var", "value"), Input("mapa-year", "value"))
def actualizar_mapa(variable, anio):
    data = df_mundial[df_mundial["year"] == anio]
    fig = px.choropleth(data, locations="Country name", locationmode="country names",
                         color=variable, hover_name="Country name", color_continuous_scale="Viridis")
    fig.update_layout(title=f"{variable} por país - {anio}", margin=dict(l=0, r=0, t=40, b=0))
    return fig


# Modelo ML

@app.callback(
    Output("pred-resultado", "children"),
    Input("pred-boton", "n_clicks"),
    State("pred-continente", "value"),
    *[State(f"pred-{i}", "value") for i in range(len(FEATURE_INPUT_COLS))],
    prevent_initial_call=True,
)
def predecir(n_clicks, continente, *valores):
    if any(v is None for v in valores):
        return "Por favor completa todos los campos."

    continente_cod = continent_encoder.transform([continente])[0]
    fila = {"Continent": continente_cod}
    fila.update(dict(zip(FEATURE_INPUT_COLS, valores)))

    entrada = pd.DataFrame([fila])[FEATURE_COLS]
    entrada_esc = scaler.transform(entrada.values)
    prediccion = modelo.predict(entrada_esc)[0]

    return f"Life Ladder estimado: {prediccion:.2f}"

if __name__ == "__main__":
    app.run(debug=True)
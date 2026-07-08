import pandas as pd
import numpy as np
import country_converter as coco
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from dash import Dash, dcc, html, Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import copy

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
slider_cols = ["Perceptions of corruption", "Positive affect", "Negative affect"]

alto_fig = 350  # alto compacto para que las gráficas no queden gigantes

PALETA_PASTEL = ["#A8D8C9", "#F4B6AE", "#B8C6E8", "#F4E1A1", "#D9BFE0", "#F7C9A3"]
MUNDIAL_COLORS = {c: PALETA_PASTEL[i % len(PALETA_PASTEL)] for i, c in enumerate(continents)}
AMERICA_COLORS = {r: PALETA_PASTEL[i % len(PALETA_PASTEL)] for i, r in enumerate(regiosen)}
CORR_COLORSCALE = [[0, "#F4B6AE"], [0.5, "#FBF7F0"], [1, "#9ED8C4"]]

grid_style = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr",
    "columnGap": "40px",
    "rowGap": "25px",
}

pred_grid_style = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr",
    "columnGap": "40px",
    "rowGap": "15px",
}

card_style = {
    "backgroundColor": "#FFFDF9",
    "borderRadius": "20px",
    "padding": "22px",
    "boxShadow": "0 4px 14px rgba(180,160,140,0.18)",
}

boton_style = {
    "backgroundColor": "#D9BFA8",
    "border": "none",
    "borderRadius": "12px",
    "padding": "10px 26px",
    "color": "#5C4A3F",
    "fontFamily": "Quicksand, sans-serif",
    "fontWeight": "600",
    "fontSize": "15px",
    "cursor": "pointer",
    "marginTop": "18px",
}

resultado_style = {
    "marginTop": "16px",
    "fontSize": "18px",
    "fontWeight": "600",
    "color": "#6B5B53",
}


def estilizar_grafico(fig, titulo, height=alto_fig):
    fig.update_layout(
        title=titulo,
        height=height,
        template="simple_white",
        paper_bgcolor="#FFFDF9",
        plot_bgcolor="#FFFDF9",
        font=dict(family="Quicksand, sans-serif", color="#5C5552", size=13),
        title_font=dict(family="Quicksand, sans-serif", size=17, color="#4A4441"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=45, r=25, t=55, b=40),
    )
    fig.update_xaxes(gridcolor="#F0E9DF", zerolinecolor="#F0E9DF")
    fig.update_yaxes(gridcolor="#F0E9DF", zerolinecolor="#F0E9DF")
    return fig


def hex_a_rgba(color_hex, alpha):
    color_hex = color_hex.lstrip("#")
    r, g, b = tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def grafico_densidad_suave(data, variable, color_col, mapa_colores):
    fig = go.Figure()
    for categoria in mapa_colores:
        muestra = data.loc[data[color_col] == categoria, variable].dropna()
        if len(muestra) < 2 or muestra.nunique() < 2:
            continue
        kde = gaussian_kde(muestra)
        xs = np.linspace(muestra.min(), muestra.max(), 200)
        ys = kde(xs)
        color = mapa_colores[categoria]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=str(categoria),
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=hex_a_rgba(color, 0.30),
        ))
    fig.update_layout(xaxis_title=variable, yaxis_title="densidad")
    return fig


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');
body {
    background-color: #F5EFE6;
    font-family: 'Quicksand', sans-serif;
    color: #5C5552;
    margin: 0;
}
h1 {
    color: #6B5B53;
    font-weight: 700;
}
h3 {
    color: #7A6A62;
    font-weight: 600;
}
label {
    color: #8A7A6F;
    font-weight: 600;
}
.tab {
    background-color: #F0E6D8 !important;
    border: none !important;
    color: #8A7A6F !important;
    font-family: 'Quicksand', sans-serif !important;
    font-weight: 600 !important;
}
.tab--selected {
    background-color: #FFFDF9 !important;
    color: #5C4A3F !important;
    border-top: 3px solid #D9BFA8 !important;
    border-bottom: none !important;
}
"""

INDEX_STRING = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>""" + CUSTOM_CSS + """</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


# Dashboard
app = Dash(__name__)
app.title = "World Happiness Report - Dashboard - Proyecto Final Analisis de Datos y Toma de Decisiones"
app.index_string = INDEX_STRING
server = app.server  # necesario para desplegar en Render/Heroku

app.layout = html.Div(style={"backgroundColor": "#F5EFE6", "padding": "30px 50px", "minHeight": "100vh"}, children=[
    html.H1("World Happiness Report - Dashboard"),

    # Pestanas
    dcc.Tabs([
        # Por Continentetes
        dcc.Tab(label="Mundial", children=[
            html.Div(style=grid_style, children=[

                html.Div(style=card_style, children=[
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
                ]),

                html.Div(style=card_style, children=[
                    html.H3("Relación entre dos variables"),
                    html.Label("Variable X"),
                    dcc.Dropdown(id="mun-scatter-x", options=var_options, value="Log GDP per capita", clearable=False),
                    html.Label("Variable Y"),
                    dcc.Dropdown(id="mun-scatter-y", options=var_options, value="Life Ladder", clearable=False),
                    dcc.Graph(id="mun-scatter-graph"),
                ]),

                html.Div(style=card_style, children=[
                    html.H3("Matriz de correlación"),
                    html.Label("Continente"),
                    dcc.Dropdown(
                        id="mun-corr-continente",
                        options=[{"label": "Todos", "value": "Todos"}] + [{"label": c, "value": c} for c in continents],
                        value="Todos", clearable=False,
                    ),
                    dcc.Graph(id="mun-corr-graph"),
                ]),

                html.Div(style=card_style, children=[
                    html.H3("Evolución temporal de Life Ladder"),
                    html.Label("Continentes a mostrar"),
                    dcc.Checklist(
                        id="mun-evo-checklist",
                        options=[{"label": c, "value": c} for c in continents],
                        value=continents, inline=True,
                    ),
                    dcc.Graph(id="mun-evo-graph"),
                ]),

            ]),
        ]),

        # America
        dcc.Tab(label="América", children=[
            html.Div(style=grid_style, children=[

                html.Div(style=card_style, children=[
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
                ]),

                html.Div(style=card_style, children=[
                    html.H3("Relación entre dos variables"),
                    html.Label("Variable X"),
                    dcc.Dropdown(id="am-scatter-x", options=var_options, value="Social support", clearable=False),
                    html.Label("Variable Y"),
                    dcc.Dropdown(id="am-scatter-y", options=var_options, value="Life Ladder", clearable=False),
                    dcc.Graph(id="am-scatter-graph"),
                ]),

                html.Div(style=card_style, children=[
                    html.H3("Matriz de correlación"),
                    html.Label("Región"),
                    dcc.Dropdown(
                        id="am-corr-region",
                        options=[{"label": "Todas", "value": "Todos"}] + [{"label": r, "value": r} for r in regiosen],
                        value="Todos", clearable=False,
                    ),
                    dcc.Graph(id="am-corr-graph"),
                ]),

                html.Div(style=card_style, children=[
                    html.H3("Evolución temporal de Life Ladder"),
                    html.Label("Regiones a mostrar"),
                    dcc.Checklist(
                        id="am-evo-checklist",
                        options=[{"label": r, "value": r} for r in regiosen],
                        value=regiosen, inline=True,
                    ),
                    dcc.Graph(id="am-evo-graph"),
                ]),

            ]),
        ]),

        # Mapa
        dcc.Tab(label="Mapa", children=[
            html.Div(style=card_style, children=[
                html.H3("Mapa mundial interactivo"),
                html.P("Pendiente."),
            ]),
        ]),

        # Modelo de ML
        dcc.Tab(label="Predicción", children=[
            html.Div(style=card_style, children=[
                html.H3("Predicción de Life Ladder"),
                html.P("Ingresa los valores para estimar el nivel de felicidad predicho por el modelo."),

                html.Label("Continente"),
                dcc.Dropdown(id="pred-continente", options=[{"label": c, "value": c} for c in continents],
                             value=continents[0], clearable=False),

                html.Div(style=pred_grid_style, children=[
                    html.Div([
                        html.Label(col),
                        dcc.Slider(id=f"pred-{i}", min=0, max=1, step=0.01,
                                   value=round(float(df[col].median()), 2),
                                   tooltip={"placement": "bottom", "always_visible": False})
                        if col in slider_cols else
                        dcc.Input(id=f"pred-{i}", type="number", value=round(float(df[col].median()), 2), step=0.01),
                    ])
                    for i, col in enumerate(input_cols)
                ]),

                html.Button("Predecir", id="pred-boton", n_clicks=0, style=boton_style),
                html.Div(id="pred-resultado", style=resultado_style),
            ]),
        ]),
    ]),
])

# Callbacks

# Mundial
@app.callback(Output("mun-dist-graph", "figure"), Input("mun-dist-var", "value"), Input("mun-dist-tipo", "value"))
def actualizar_dist_mundial(variable, tipo):
    if tipo == "densidad":
        fig = grafico_densidad_suave(df_mundial, variable, "Continent", MUNDIAL_COLORS)
    else:
        fig = px.violin(df_mundial, x="Continent", y=variable, color="Continent",
                         color_discrete_map=MUNDIAL_COLORS, box=True)
    return estilizar_grafico(fig, f"Distribución de {variable} por continente")


@app.callback(Output("mun-corr-graph", "figure"), Input("mun-corr-continente", "value"))
def actualizar_corr_mundial(continente):
    data = df_mundial if continente == "Todos" else df_mundial[df_mundial["Continent"] == continente]
    corr = data[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=CORR_COLORSCALE, zmin=-1, zmax=1)
    return estilizar_grafico(fig, f"Correlación entre variables ({continente})", height=alto_fig + 100)


@app.callback(Output("mun-evo-graph", "figure"), Input("mun-evo-checklist", "value"))
def actualizar_evo_mundial(continentes_sel):
    data = df_mundial[df_mundial["Continent"].isin(continentes_sel)]
    resumen = data.groupby(["year", "Continent"], as_index=False)["Life Ladder"].mean()
    fig = px.line(resumen, x="year", y="Life Ladder", color="Continent",
                   color_discrete_map=MUNDIAL_COLORS, markers=True)
    return estilizar_grafico(fig, "Evolución de Life Ladder promedio por continente")


@app.callback(Output("mun-scatter-graph", "figure"), Input("mun-scatter-x", "value"), Input("mun-scatter-y", "value"))
def actualizar_scatter_mundial(var_x, var_y):
    fig = px.scatter(df_mundial, x=var_x, y=var_y, color="Continent",
                      color_discrete_map=MUNDIAL_COLORS, opacity=0.7)
    return estilizar_grafico(fig, f"{var_y} vs {var_x}")


# America
@app.callback(Output("am-dist-graph", "figure"), Input("am-dist-var", "value"), Input("am-dist-tipo", "value"))
def actualizar_dist_america(variable, tipo):
    if tipo == "densidad":
        fig = grafico_densidad_suave(df_america, variable, "Region", AMERICA_COLORS)
    else:
        fig = px.violin(df_america, x="Region", y=variable, color="Region",
                         color_discrete_map=AMERICA_COLORS, box=True)
    return estilizar_grafico(fig, f"Distribución de {variable} por región de América")


@app.callback(Output("am-corr-graph", "figure"), Input("am-corr-region", "value"))
def actualizar_corr_america(region):
    data = df_america if region == "Todos" else df_america[df_america["Region"] == region]
    corr = data[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=CORR_COLORSCALE, zmin=-1, zmax=1)
    return estilizar_grafico(fig, f"Correlación entre variables ({region})", height=alto_fig + 100)


@app.callback(Output("am-evo-graph", "figure"), Input("am-evo-checklist", "value"))
def actualizar_evo_america(regiones_sel):
    data = df_america[df_america["Region"].isin(regiones_sel)]
    resumen = data.groupby(["year", "Region"], as_index=False)["Life Ladder"].mean()
    fig = px.line(resumen, x="year", y="Life Ladder", color="Region",
                   color_discrete_map=AMERICA_COLORS, markers=True)
    return estilizar_grafico(fig, "Evolución de Life Ladder promedio por región de América")


@app.callback(Output("am-scatter-graph", "figure"), Input("am-scatter-x", "value"), Input("am-scatter-y", "value"))
def actualizar_scatter_america(var_x, var_y):
    fig = px.scatter(df_america, x=var_x, y=var_y, color="Region",
                      color_discrete_map=AMERICA_COLORS, opacity=0.7)
    return estilizar_grafico(fig, f"{var_y} vs {var_x}")


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
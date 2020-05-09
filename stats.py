import requests
import json

import pandas as pd

import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc

import plotly.graph_objects as go

from datetime import date
from datetime import timedelta

from pmdarima import auto_arima

external_scripts = ['/assets/style.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]

# Fetching Data

url = "https://api.covid19india.org/data.json"
payload = {}
headers = {
    'Content-Type': 'application/json'
}

response = requests.request("GET", url, headers=headers, data=payload)

json_data = json.loads(response.text)

# converting data into data frame

daily_cases = json_data['cases_time_series']
statewise = json_data['statewise']

deceased = daily_cases[-1]["totaldeceased"]
recovered = daily_cases[-1]["totalrecovered"]
active = int(daily_cases[-1]["totalconfirmed"]) - int(recovered) - int(deceased)

dates = []
confirmed = []
future = []

# data frame columns
for i in range(0, len(daily_cases)):
    dates.append(daily_cases[i]['date'])
    confirmed.append(daily_cases[i]['totalconfirmed'])

lists = {
    'Date': dates,
    'Cases': confirmed
}

df = pd.DataFrame(lists)

df = df.set_index('Date')
df.index.freq = 'D'
df['Cases'] = df['Cases'].astype('float32')

zipped = list(zip(dates, confirmed))
df = pd.DataFrame(zipped, columns=['Date', 'Cases'])

# Dates for next 7 days
for i in range(0, 8):
    future.append(str(date.today() + timedelta(days=i)))

# ARIMA Model

stepwise_model = auto_arima(df['Cases'], start_p=1, start_q=1,
                            max_p=5, max_q=5, seasonal=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
model = stepwise_model.fit(df['Cases'])

future_forecast = stepwise_model.predict(n_periods=7)
future_forecast = np.ceil(future_forecast)

zipped_forecast = list(zip(future, future_forecast))
df_future = pd.DataFrame(zipped_forecast, columns=['Date', 'Cases'])

observed_yesterday = daily_cases[-1]['totalconfirmed']
expected = int(future_forecast[0]) - int(observed_yesterday)

# Graphs depicting Daily Changes

daily_dates = []
daily_recovered = []
daily_deceased = []
daily_active = []

for i in range((len(daily_cases) - 30), len(daily_cases)):
    daily_dates.append(daily_cases[i]['date'])
    daily_recovered.append(daily_cases[i]['dailyrecovered'])
    daily_deceased.append(daily_cases[i]['dailydeceased'])
    daily_active.append(int(daily_cases[i]['dailyconfirmed']) - int(daily_cases[i]['dailyrecovered']) - int(
        daily_cases[i]['dailydeceased']))

colors_length = len(daily_dates)

# daily changes

colors_confirmed = ['crimson', ] * colors_length
colors_active = ['blue', ] * colors_length
colors_recovered = ['green', ] * colors_length
colors_deceased = ['grey', ] * colors_length

fig_forecast = go.Figure(data=[
    go.Bar(name='Confirmed', x=future, y=future_forecast,
           text=future_forecast, textposition='auto',
           marker_color=colors_confirmed)
])

fig_active = go.Figure(data=[
    go.Bar(name='Active', x=daily_dates, y=daily_active,
           marker_color=colors_active)
])

fig_recovered = go.Figure(data=[
    go.Bar(name='Recovered', x=daily_dates, y=daily_recovered,
           marker_color=colors_recovered)
])

fig_deceased = go.Figure(data=[
    go.Bar(name='Deceased', x=daily_dates, y=daily_deceased,
           marker_color=colors_deceased)
])

# projections tab

card_daily = dbc.Container([dbc.Row([
    dbc.Col([
        dbc.CardHeader(html.H5("Next 7 Days Forecast"), ),
        dbc.Card([dcc.Graph(figure=fig_forecast, )])], sm=12, md=12, lg=6, xl=6,
    ),
    dbc.Col([
        dbc.CardHeader(html.H5("Daily Active Cases"), ),
        dbc.Card([dcc.Graph(figure=fig_active)])], sm=12, md=12, lg=6, xl=6,
    ),
]),
    dbc.Row([
        dbc.Col([
            dbc.CardHeader(html.H5("Daily Recovered Cases"), ),
            dbc.Card(dcc.Graph(figure=fig_recovered)),
        ], sm=12, md=12, lg=6, xl=6, ),
        dbc.Col([
            dbc.CardHeader(html.H5("Daily Deceased Cases"), ),
            dbc.Card(dcc.Graph(figure=fig_deceased)),
        ], sm=12, md=12, lg=6, xl=6, ),
    ])
])

header = dbc.Container(
    html.H1("Covid 19 Metrics Distributions",
            style={"textAlign": "center"})
)

footer = html.Div(
    [
        dbc.Alert(
            [
                "Open Sourced At ",
                html.A("GitHub", href="https://github.com/Stephen2206/predicting-covid19-cases-India",
                       className="alert-link"),
                html.Br(),
                html.A("Crowd Sourced Patient Database", href="https://api.covid19india.org/", className="alert-link"),
            ],
            color="primary",
        ),
    ]
)

alerts = dbc.Container(
    [
        dbc.Alert("NOTE:This is just a predictive model. Actual Results may vary.", color="danger",
                  style={"textAlign": "center"}),
    ]
)

# Table for 7 days forecast
table = dbc.Table.from_dataframe(df_future, striped=True, bordered=True, hover=True)

today_confirmed = statewise[0]["confirmed"]

delta_confirmed = str(statewise[0]["deltaconfirmed"])
delta_recovered = str(statewise[0]["deltarecovered"])
delta_deceased = str(statewise[0]["deltadeaths"])
# bar graph holder

cards_metrics = dbc.Container([dbc.Row([
    dbc.Col(dbc.Card([html.H6("Cases Expected Today "),
                      html.H6(str(int(future_forecast[0])) +
                              str('[+' + str(expected) + ']')),
                      ],
                     color="primary", inverse=True), xs=6, sm=6, md=6, lg=3, xl=3
            ),
    dbc.Col(dbc.Card([html.H6("Confirmed Cases "),
                      html.H6(str(int(today_confirmed)) +
                              "[+" + delta_confirmed + "]"),
                      ],
                     color="danger", inverse=True), xs=6, sm=6, md=6, lg=3, xl=3
            ),
    dbc.Col(dbc.Card([html.H6("Recovered "),
                      html.H6(str(int(statewise[0]["recovered"])) +
                              "[+" + delta_recovered + "]"),
                      ],
                     color="success", inverse=True), xs=6, sm=6, md=6, lg=3, xl=3
            ),
    dbc.Col(dbc.Card([html.H6("Deceased "),
                      html.H6(str(int(statewise[0]["deaths"])) +
                              "[+" + delta_deceased + "]"),
                      ],
                     color="warning", inverse=True), xs=6, sm=6, md=6, lg=3, xl=3
            ),
]),
])

# Ststewise Count

states = []
state_active = []
state_confirmed = []
state_recovered = []
state_deaths = []

# state_active_daily = []
state_confirmed_daily = []
state_recovered_daily = []
state_deaths_daily = []

for i in statewise:
    states.append(i['state'])
    state_active.append(i['active'])
    state_confirmed.append(i['confirmed'])
    state_deaths.append(i['deaths'])
    state_recovered.append(i['recovered'])

    state_confirmed_daily.append(i['deltaconfirmed'])
    state_deaths_daily.append(i['deltadeaths'])
    state_recovered_daily.append(i['deltarecovered'])

df_states_daily = pd.DataFrame(list(zip(states, state_confirmed_daily, state_recovered_daily, state_deaths_daily)),
                               columns=['State', 'Cnfrmd', 'Rcvrd', 'Dths'])
df_states = pd.DataFrame(list(zip(states, state_confirmed,state_active,state_recovered,state_deaths)),
                         columns=['State', 'Cnfrmd', 'Active', 'Rcvrd', 'Dths'])

table_state = dbc.Table.from_dataframe(df_states, striped=True, bordered=True, hover=True, responsive=True, size='sm')

table_state_daily = dbc.Table.from_dataframe(df_states_daily, striped=True, bordered=True, hover=True, responsive=True,size='sm')


# embedding into tabs
tab1_content = dbc.Card(
    dbc.CardBody(
        [
            # html.P("Cases Today", className="card-text"),
            table_state_daily,
        ]
    ),
    className="mt-3",
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            # html.P("Total Cases", className="card-text"),
            table_state,
        ]
    ),
    className="mt-3",
)


tabs_state = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Cases Today"),
        dbc.Tab(tab2_content, label="Total Cases"),
    ]
)


# app starts

covid = dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets,
                  meta_tags=[
                      {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                  ]
                  )
server = covid.server
covid.title = "COVID-19 PREDICTIONS India"

# layout
covid.layout = html.Div([
    dcc.Tabs(id="tabs", children=[
        # Left Tab
        dcc.Tab(label='Dashboard', children=[
            html.Div([
                html.Div([header,
                          cards_metrics,
                          html.Br(),
                          alerts,
                          dbc.Container([
                              dbc.Row([dbc.Col(
                                  [
                                      dbc.CardHeader(html.H5("Predictions for Next Week")),
                                      dbc.Card(table),
                                  ], xs=12, sm=12, md=12, lg=12, xl=12),
                              ], ),
                              html.Br(),
                              tabs_state,
                          ]),

                          ]),

                footer]),
        ]),
        # Right Tab
        dcc.Tab(label='Cases Projections', children=[
            html.Br(),
            html.Div([
                card_daily,
                footer
            ], )
        ])  # end of right tab
    ])  # tabs end
])

# layout ends

if __name__ == '__main__':
    covid.run_server(debug=True)

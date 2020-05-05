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

# Fetching Data Till Yesterday

url = "https://api.covid19india.org/data.json"
payload = {}
headers = {
    'Content-Type': 'application/json'
}

response = requests.request("GET", url, headers=headers, data=payload)

json_data = json.loads(response.text)

# converting data into data frame

daily_cases = json_data['cases_time_series']
state_wise = json_data['statewise']

deceased = daily_cases[-1]["totaldeceased"]
recovered = daily_cases[-1]["totalrecovered"]
active = int(daily_cases[-1]["totalconfirmed"]) - int(recovered) - int(deceased)

dates = []
confirmed = []
future = []

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

'''


length_df = len(df)
length_train = (length_df-7)
length_test = 7
train_data = df.iloc[:length_train,:]
test_data = df.iloc[length_train:,:]

stepwise_model = auto_arima(train_data['Cases'], start_p=1, start_q=1,
                           max_p=5, max_q=5,seasonal=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
model = stepwise_model.fit(train_data['Cases'])

test_forecast = stepwise_model.predict(n_periods=length_test)
test_forecast = np.ceil(test_forecast)

dates = test_data['Date']

fig_model = go.Figure(data=[
    go.Bar(name='Predicted', x=dates, y=test_forecast),
    go.Bar(name='Observed', x=dates, y=test_data['Cases'])
])
fig_model.update_layout(barmode='group')
'''

'''Dates for next 7 days'''
for i in range(0, 8):
    future.append(str(date.today() + timedelta(days=i)))

'''ARIMA Model'''

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
    go.Bar(name='Confirmed', x=daily_dates, y=future_forecast,
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

header = html.H1("Covid 19 Metrics Distributions", style={"textAlign": "center"})

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

alerts = html.Div(
    [
        dbc.Alert("NOTE:This is just a predictive model. Actual Results may vary.", color="danger",
                  style={"textAlign": "center"}),
    ]
)

# Table for 7 days forecast
table = dbc.Table.from_dataframe(df_future, striped=True, bordered=True, hover=True)

today_confirmed = state_wise[0]["confirmed"]

# Cards depicting Cases Status


# bar graph holder
'''
last_week = dbc.Row([dbc.Col(dbc.Card(
    [dbc.CardHeader(html.H5("Last Week Results"),style={"text-align":"center"}),
     dcc.Graph(figure=fig_model)])),
])
'''
cards_metrics = html.Div([dbc.Row([
    dbc.Col(dbc.Card([html.H5("Cases Expected Today "),
                      html.H5(str(future_forecast[0])+
                              str('[+' + str(expected) + ']')),
                      ],
                     color="primary", inverse=True), xs=6, sm=6, md=6, lg=3, xl=3
            ),
    dbc.Col(dbc.Card([html.H5("Confirmed Cases "),
                      html.H5(str(int(today_confirmed))+
                              "[+" + str(state_wise[0]["deltaconfirmed"]) + "]"),
                      ],
                     color="danger", inverse=True), xs=6, sm=6, md=6, lg=3, xl=3
            ),
    dbc.Col(dbc.Card([html.H5("Recovered "),
                      html.H5(str(int(state_wise[0]["recovered"]))+
                              "[+" + str(state_wise[0]["deltarecovered"]) + "]"),
                      ],
                     color="success", inverse=True), xs=6, sm=6, md=6, lg=3, xl=3
            ),
    dbc.Col(dbc.Card([html.H5("Deceased "),
                      html.H5(str(int(state_wise[0]["deaths"])) +
                              "[+" + str(state_wise[0]["deltadeaths"]) + "]"),
                      ],
                     color="warning", inverse=True), xs=6, sm=6, md=6, lg=3, xl=3
            ),
]),
])

covid = dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets)
server = covid.server
covid.title = "COVID-19 PREDICTIONS India"

covid.layout = html.Div([
    dcc.Tabs(id="tabs", children=[
        # Left Tab
        dcc.Tab(label='Dashboard', children=[
            dbc.Container([header,
                           cards_metrics,
                           html.Br(),
                           alerts,
                           html.Div(
                               dbc.Row([dbc.Col(
                                   [
                                       dbc.CardHeader(html.H5("Predictions for Next Week")),
                                       dbc.Card(table),
                                   ], sm=8, md=8, lg=8,
                               ),
                               ],),
                           ),
                           dbc.Container([
                               # last_week,
                               html.Br(),
                           ]),
                           ]),
            footer
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

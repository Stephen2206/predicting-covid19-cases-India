import requests
import json

import pandas as pd

import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go

from datetime import date
from datetime import timedelta

from pmdarima import auto_arima

external_scripts = ['/assets/style.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]

'''Fetching Data Till Yesterday'''

url = "https://api.covid19india.org/data.json"
payload = {}
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("GET", url, headers=headers, data=payload)

json_data = json.loads(response.text)


'''converting data into data frame'''

daily_cases = json_data['cases_time_series']
dates = []
confirmed = []
future = []


deceased = daily_cases[-1]["totaldeceased"]
recovered = daily_cases[-1]["totalrecovered"]
active = int(daily_cases[-1]["totalconfirmed"])-int(recovered)-int(deceased)

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


'''Split dataset into train and test'''

length_df = len(df)
length_train = (length_df-7)
length_test = 7
train_data = df.iloc[:length_train,:]
test_data = df.iloc[length_train:,:]

'''Model For Test Data'''
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


'''Dates for next 7 days'''
for i in range(0, 8):
    future.append(str(date.today()+timedelta(days=i)))


'''ARIMA Model'''

stepwise_model = auto_arima(df['Cases'], start_p=1, start_q=1,
                           max_p=5, max_q=5,seasonal=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
model = stepwise_model.fit(df['Cases'])

future_forecast = stepwise_model.predict(n_periods=7)
future_forecast = np.ceil(future_forecast)

zipped_forecast = list(zip(future, future_forecast))
df_future = pd.DataFrame(zipped_forecast, columns=['Date', 'Cases'])
fig_forecast = px.line(df_future, x="Date", y="Cases")

observed_yesterday = daily_cases[-1]['totalconfirmed']
expected = (int)(future_forecast[0])-int(observed_yesterday)

# Graphs depicting Daily Changes

daily_dates = []
daily_confirmed = []
daily_recovered = []
daily_deceased = []
daily_active = []

for i in range(50,len(daily_cases)):
    daily_dates.append(daily_cases[i]['date'])
    daily_confirmed.append(daily_cases[i]['dailyconfirmed'])
    daily_recovered.append(daily_cases[i]['dailyrecovered'])
    daily_deceased.append(daily_cases[i]['dailydeceased'])
    daily_active.append(int(daily_cases[i]['dailyconfirmed'])-int(daily_cases[i]['dailyrecovered'])-int(daily_cases[i]['dailydeceased']))


colors_length = len(daily_dates)

# daily changes
colors_confirmed = ['crimson', ] * colors_length
colors_active = ['blue', ] * colors_length
colors_recovered = ['green',] * colors_length
colors_deceased = ['grey',] * colors_length

fig_confirmed = go.Figure(data=[
    go.Bar(name='Confirmed', x=daily_dates, y=daily_confirmed,
           marker_color = colors_confirmed)
])

fig_active = go.Figure(data=[
    go.Bar(name='Active', x=daily_dates, y=daily_confirmed ,
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


card_daily = html.Div([dbc.Row([
    dbc.Col([
        dbc.CardHeader(html.H5("Daily Confirmed Cases"),),
        dbc.Card([dcc.Graph(figure=fig_confirmed,)])], width=6
    ),
    dbc.Col([
        dbc.CardHeader(html.H5("Daily Active Cases"),),
        dbc.Card([dcc.Graph(figure=fig_active)])],width=6
    ),
]),
    dbc.Row([
        dbc.Col([
            dbc.CardHeader(html.H5("Daily Recovered Cases"),),
            dbc.Card(dcc.Graph(figure=fig_recovered)),
        ],width=6),
        dbc.Col([
            dbc.CardHeader(html.H5("Daily Deceased Cases"),),
            dbc.Card(dcc.Graph(figure=fig_deceased)),
        ], width=6),
    ])
])


header = html.H1("Covid 19 Metrics Distributions", style={"textAlign": "center"})

alerts = html.Div(
    [
        dbc.Alert("NOTE:This is just a predictive model. Actual Results may vary.", color="danger",style={"textAlign":"center"}),
    ]
)

# Table for 7 days forecast
table = dbc.Table.from_dataframe(df_future, striped=True, bordered=True, hover=True)

# Cards depicting Cases Status
cards_metrics = html.Div([dbc.Row([
    dbc.Col(dbc.Card([html.H5("Cases Expected Today:"),
                      html.H5(str(future_forecast[0])+str('(+'+str(expected)+')'))],
                     color="primary", inverse=True)
            ),
    dbc.Col(dbc.Card([html.H5("Confirmed Cases :"),
                      html.H5(str(observed_yesterday))], color="danger", inverse=True)
            ),
    dbc.Col(dbc.Card([html.H5("Recovered :"),
                      html.H5(str(recovered))], color="success", inverse=True)
            ),
    dbc.Col(dbc.Card([html.H5("Deceased :"),
                      html.H5(str(deceased))], color="warning", inverse=True)
            ),
], className="mb-3",),
])

# bar graph holder
last_week = dbc.Row([dbc.Col(dbc.Card(
    [dbc.CardHeader(html.H5("Last Week Results"),style={"text-align":"center"}),
     dcc.Graph(figure=fig_model)])),
])

covid = dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets)
server = covid.server

covid.layout = html.Div([
    html.Title("COVID-19 Predictions"),
    dcc.Tabs(id="tabs", children=[
        # Left Tab
        dcc.Tab(label='Dashboard', children=[
            html.Div([header,
                      cards_metrics,
                      html.Br(),
                      alerts,
                      html.Div(

                          dbc.Row([dbc.Col(
                              [
                               dbc.CardHeader(html.H5("Predictions for Next Week")),
                               ], width=12,
                          ),
                          ]),
                      ),
                      html.Div(
                          dbc.Row([dbc.Col([
                              dbc.Card(table),
                          ], width=5),
                              dbc.Col([dbc.Card(dcc.Graph(figure=fig_forecast))])
                          ])
                      ),
                      html.Div([

                          last_week,
                          html.Br(),
                      ]),
                      ], className="container"),
        ]),
        # Right Tab
        dcc.Tab(label='Daily Projections', children=[
            html.Br(),
            html.Div([
                card_daily
            ], className="container")
        ])  # end of right tab
        ])  # tabs end
])

# layout ends


if __name__ == '__main__':
    covid.run_server(debug=True)

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 07:57:02 2020
@author: Chen Shen
"""

#%% Load library
import numpy as np
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash
import time

#%% Run workers
import utility
from rq import Queue
from worker import conn

q = Queue(connection=conn)

#%%
job_getfit = q.enqueue(utility.getfit, t1='2020-01-01',t2='2020-12-31')

t0 = time.time()
while job_getfit.result is None:
    t1 = time.time()
    t2 = t1-t0
    time.sleep(5)
    print('waiting: {}'.format(t2))

print('Finished! Time elapse: {}'.format(t2))
df_getfit = job_getfit.result

#%%
import importlib
importlib.reload(utility)
import matplotlib.pyplot as plt
df_yc = utility.ycnsresult(
    df_getfit['t_cal'],
    df_getfit['fit_par'])
f, fbs = utility.graph(df_getfit['t_cal'][:-1],
                       df_yc[:-1],
                       df_getfit['fit_par'][:-1])

#%%
yp =df_getfit['fit_par'][:-1][2] 

def lag_diff(x,lag):
    x_shift = np.append(np.zeros(lag),x)
    x_diff = np.append(x,np.zeros(lag))-x_shift
    return np.append(np.zeros(lag),x_diff[lag:-lag])

x = lag_diff(yp,1)
i_x = np.where(x > (x.mean()+3*x.std()))

plt.plot(yp)
plt.scatter(i_x,yp.iloc[i_x],color='red')

#%% Dash

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Yield-Curve Dashboard", href="#")),
        dbc.NavItem(dbc.NavLink("Shock Analytics", href="#")),
        dbc.NavItem(dbc.NavLink("Appendix", href="#")),
    ],
    brand="Demo - US Daily Treasury Curve Modelling",
    brand_href="#",
    sticky="top",
)

body = dbc.Container(
    [
        dbc.Row(
            dbc.Row(
                [
                    dbc.Col(
                        [

                            dcc.Graph(id='graph', figure=f)
                        ], width=6
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id='Beta_0', figure=fbs,
                            )
                        ], width=4
                    ),
                ], no_gutters=True
            )),
        dbc.Row(
            [
                html.Div(id='Selected_time', children=''),
                dbc.Col(
                    dcc.Slider(
                        id='time_i',
                        min=0,
                        max=len(df_getfit['t_cal'])-1,
                        step=1,
                        value=0
                    ), width=5
                )                
            ]
        ),
        dbc.Row(
            [
                html.A(
                    id='link',
                    children='',
                    href=''
                )
            ],
            justify="center"
        )
    ]
)
app.layout = html.Div([navbar, body])


@app.callback(
    [Output(component_id='graph', component_property='figure'),
     Output(component_id='Beta_0', component_property='figure'),
     Output(component_id='Selected_time', component_property='children')],
    [Input(component_id='time_i', component_property='value')]
)
def update_figure(time_index):
    fit_yc_new = df_yc[time_index]
    t = df_getfit['t_cal'].iloc[time_index]
    tfit = np.linspace(0, 30, 50)

    with f.batch_update():
        f.data[1].z = fit_yc_new
        f.data[1].x = np.repeat(t, tfit.shape[0])
        #print('success')
    with fbs.batch_update():
        fbs.data[9].x = [t, t]
        fbs.data[10].x = [t, t]
        fbs.data[11].x = [t, t]
        #print('success')
    return f, fbs, 'Time Slider: <' + t.strftime('%b %d %Y') + '>'

if __name__ == "__main__":
	app.run_server(debug=True)

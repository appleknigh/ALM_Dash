# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 07:56:06 2020

@author: 330411836
"""

#%%
import pandas as pd
import numpy as np
from nelson_siegel_svensson import NelsonSiegelCurve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from CSModules import ALM_kit, nssmodel
import quandl
import copy

#%%
def getfit(t1='2000-01-02', t2='2020-12-02'):
    #Load data
    df_YieldC = quandl.get(
        "USTREASURY/YIELD", authtoken="4_zrDSANo7hMt_uhyzQy")
    df_YieldC.reset_index(level=0, inplace=True)
    df_YieldC['Date'] = pd.to_datetime(df_YieldC['Date'], format="%m/%d/%Y")

    #NS Cure fit
    t_cal = df_YieldC['Date']
    i_range = np.where((t_cal > t1) & (t_cal < t2))

    t = np.array([0.08333333, 0.16666667, 0.25,
                  0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    y = np.array(df_YieldC.iloc[:, 1:]).astype(float)[i_range]
    fit_par = pd.DataFrame(np.apply_along_axis(
        lambda x: ALM_kit.ns_par(t, x), axis=1, arr=y))
    return {'df_YieldC': df_YieldC, 't_cal': t_cal.iloc[i_range], 'tact': t, 'y': y, 'fit_par': fit_par}


def ycnsresult(calendar_t, fit_par, tfit=np.linspace(0, 30, 50)):
    s_z = np.array([ALM_kit.yfit_beta(tfit, fit_par.iloc[i])
                    for i in range(fit_par.shape[0])])
    return s_z


def graph(calendar_t, s_z, fit_par,tfit=np.linspace(0, 30, 50)):
    #Setting
    color_scale = [[0, "rgb(31, 119, 180)"], [1, "rgb(31, 119, 180)"]]
    light_effect = dict(
        fresnel=0.01,
        specular=0.01,
        ambient=0.95,
        diffuse=0.99,
        roughness=0.01
    )
    camera = dict(
        eye=dict(x=2.5, y=0., z=0.)
    )

    #Historical YC surface plot
    f = go.Figure()
    f.add_surface(x=calendar_t, y=tfit, z=s_z.transpose(),
                  colorscale=color_scale, opacity=0.2, showscale=False)
    f.add_scatter3d(x=np.repeat(
        calendar_t.iloc[0], tfit.shape[0]
    ), y=tfit, z=s_z[0], mode='lines', line_color="rgb(0, 0, 0)", line_width=3)

    camera = dict(
        eye=dict(x=1.25, y=1.25, z=1.25)
    )

    f.update_layout(
        scene=dict(
            xaxis=dict(title_text='Calendar time'),
            yaxis=dict(title_text='Term'),
            zaxis=dict(title_text='Yield')),
        scene_camera=camera,
        title={
            'text': "US Treasury Yield Curve",
            'x': 0.5,
            'y': 0.9},
        width=700,
        height=700
    )

    #Factors plots (level, slope, curvature)
    fbs = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Level", "Slope", "Curvature")
    )

    fbs.add_scatter(x=calendar_t, y=fit_par.iloc[:, 0], row=1, col=1)
    fbs.add_scatter(x=calendar_t, y=fit_par.iloc[:, 1], row=2, col=1)
    fbs.add_scatter(x=calendar_t, y=fit_par.iloc[:, 2], row=3, col=1)

    t1 = calendar_t.iloc[-1]
    fbs.add_scatter(x=[t1, t1], y=[fit_par.iloc[:, 0].min(),
                                   fit_par.iloc[:, 0].max()], row=1, col=1, mode='lines')
    fbs.add_scatter(x=[t1, t1], y=[fit_par.iloc[:, 1].min(),
                                   fit_par.iloc[:, 1].max()], row=2, col=1, mode='lines')
    fbs.add_scatter(x=[t1, t1], y=[fit_par.iloc[:, 2].min(),
                                   fit_par.iloc[:, 2].max()], row=3, col=1, mode='lines')

    fbs.update_layout(width=500, height=700, showlegend=False)
    return f, fbs

#%%
if __name__ == '__main__':
    df_getfit = getfit(t1='2019-01-01',t2='2020-12-31')
    df_yc = ycnsresult(df_getfit['t_cal'],df_getfit['fit_par'])
    f, fbs = graph(df_getfit['t_cal'], df_yc, df_getfit['fit_par'])


# %%

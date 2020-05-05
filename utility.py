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


def aol(yp1,yp2,n):
    slope = (yp2 - yp1)/n
    k = yp1
    t = np.linspace(1,50,50)

    m_t = np.zeros([len(t),len(t)])
    np.fill_diagonal(m_t,t)
    m_slope = np.tile(slope,(50,1))
    m_k = np.tile(k,(50,1))
    m = np.dot(m_t,m_slope) + m_k
    return m

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
            'text': "US treasury yield curve at a glance",
            'x': 0.5,
            'y': 0.9},
        width=700,
        height=700
    )

    #Factors plots (level, slope, curvature)
    fbs = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Long-term level',None,'Inversion slope', None, 'Humped curve'],
        specs=[
            [{"type": "xy",}, {"type": "table"}],
            [{"type": "xy"}, {"type": "table"}],
            [{"type": "xy"}, {"type": "table"}]
        ],
        column_widths=[1.5,1]
    )

    def anomaly(x0):
        x = np.append(0,np.diff(x0))
        i_x = np.where(x > (x.mean()+4*x.std()))
        return i_x
        
    yc_level = s_z[:, -1]
    yc_ilevel = anomaly(yc_level)
    print(calendar_t.iloc[yc_ilevel])
    print(yc_level[yc_ilevel])
    
    yc_slope = s_z[:, -1]-s_z[:, 0]
    yc_islope = anomaly(yc_slope)
    yc_curve = s_z[:, -1]-s_z[:, 16]
    yc_icurve = anomaly(yc_curve)
    
    def tseries_graph(X, Y, i, perc_h, perc_l, nrow, ncol):
        fbs.add_scatter(x=X,
                        y=np.repeat(np.quantile(Y, 0.05),X.shape[0]),
                        line_color='grey', row=nrow, col=ncol, line=dict(width=0))
        fbs.add_scatter(x=X,
                        y=np.repeat(np.quantile(Y, 0.95),X.shape[0]),
                        fill='tonexty', line_color='grey', opacity=0.5,
                        row=nrow, col=ncol, line=dict(width=0))
        fbs.add_scatter(x=X, y=Y,
                        line_color='black', row=nrow, col=ncol)
        #fbs.add_scatter(x=X.iloc[i], y=Y[i], mode='markers',
        #                row=nrow, col=ncol)

    tseries_graph(calendar_t,yc_level,yc_ilevel,0.95,0.05,1,1)
    tseries_graph(calendar_t,yc_slope,yc_islope,0.95,0.05,2,1)
    tseries_graph(calendar_t,yc_curve,yc_icurve,0.95,0.05,3,1)

    t1 = calendar_t.iloc[-1]
    fbs.add_scatter(x=[t1, t1], y=[yc_level.min()-yc_level.std()*0.5,
                                   yc_level.max()+yc_level.std()*0.5], row=1, col=1, mode='lines')
    fbs.add_scatter(x=[t1, t1], y=[yc_slope.min()-yc_slope.std()*0.5,
                                   yc_slope.max()+yc_slope.std()*0.5], row=2, col=1, mode='lines')
    fbs.add_scatter(x=[t1, t1], y=[yc_curve.min()-yc_curve.std()*0.5,
                                   yc_curve.max()+yc_curve.std()*0.5], row=3, col=1, mode='lines')
    
    def yc_x(X,nrow,ncol,i1=-1,i2=-14,i3=-30,ttext = 'Rates'):
        yc_x_d = np.round(X[-1],2)
        yc_x_biweek = np.round(X[-14:-1].mean(),2)
        yc_x_month = np.round(X[-30:-1].mean(),2)
        
        fbs.add_trace(
            go.Table(
                header=dict(
                    values=list([calendar_t.iloc[-1].strftime('%b%d'),ttext])
                ),
                cells=dict(values=[['Daily', '14d', '30d'],
                [yc_x_d,yc_x_biweek,yc_x_month]]
                ),
                ),row=nrow, col=ncol
        )

    yc_x(yc_level,1,2,ttext='Level')
    yc_x(yc_slope,2,2,ttext='Slope')
    yc_x(yc_curve,3,2,ttext='Curve')

    fbs.update_layout(width=500, height=700, showlegend=False)
    fbs.update_xaxes(tickformat='%d %b %Y')
    fbs.update_yaxes(title_text="30yr rate", row=1, col=1)
    fbs.update_yaxes(title_text="(30yr - 3mo) rate", row=2, col=1)
    fbs.update_yaxes(title_text="(30yr - 10yr) rate", row=3, col=1)
    return f, fbs

#%%
if __name__ == '__main__':
    df_getfit = getfit(t1='2019-01-01',t2='2020-12-31')
    df_yc = ycnsresult(df_getfit['t_cal'],df_getfit['fit_par'])
    f, fbs = graph(df_getfit['t_cal'], df_yc, df_getfit['fit_par'])


# %%

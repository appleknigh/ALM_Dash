#%%
i = 2
df_yc[i,:]

yp1 = df_yc[:,0]
yp2 = df_yc[:,-1]
n = df_yc.shape[1]
t = np.linspace(1,50,50)

#%%
m = utility.aol(df_yc[:,0],df_yc[:,-1],df_yc.shape[1])


#%%
plt.plot(df_yc[1,:])
plt.plot(m[:,1])

aol = np.sum(df_yc/np.transpose(m)-1,axis=1)
aol = (df_yc[:,-1]-df_yc[:,25])

#%%
plt.plot(aol)

plt.plot(df_getfit['fit_par'].iloc[:,2]*0.5,color='red')

#%%
plt.plot(-df_getfit['fit_par'].iloc[:,1])
plt.plot(df_getfit['fit_par'].iloc[:,2]*0.5+1.5,color='red')

#%%
yp = np.dot(slope,t)+yp1

plt.plot(yp)
plt.plot(df_yc[i,:])


aol = np.sum(yp-df_yc[i,:])


#%%
c = np.zeros([5,5])
np.fill_diagonal(c,b)

#%%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=3,cols=3,
specs=[
    [{"type":"table","colspan":"2"},{None},{"type":"table"}],
    [{"type":"table","colspan":"2"},{None},{"type":"table"}],
    [{"type":"table","colspan":"2"},{None},{"type":"table"}]
    ])

#%%
def yc_x(X,nrow,ncol,i1=-1,i2=-14,i3=-30):
    yc_x_d = np.round(X[-1],2)
    yc_x_biweek = np.round(X[-14:-1].mean(),2)
    yc_x_month = np.round(X[-30:-1].mean(),2)
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(['Daily', '14 days', '30 days'])
            ),
            cells=dict(values=[[yc_x_d],
                            [yc_x_biweek],
                            [yc_x_month]]),
            ),row=nrow, col=ncol
    )



yc_level = df_yc[:,-1]
yc_slope = df_yc[:,-1]-df_yc[:,0]
yc_curve = df_yc[:,-1]-df_yc[:,25]

#yc_level_d, yc_level_biweek, yc_level_month = yc_x(yc_level)
#yc_slope_d, yc_slope_biweek, yc_slope_month = yc_x(yc_slope)
#yc_curve_d, yc_curve_biweek, yc_curve_month = yc_x(yc_curve)



#%%

yc_x(yc_level,1,2)
yc_x(yc_level,2,2)
yc_x(yc_level,3,2)

#%%

fig.add_trace(
    go.Table(
        header=dict(
            values=list(['Daily', '14 days', '30 days'])
        ),
        cells=dict(values=[[yc_level_d],
                           [yc_level_biweek],
                           [yc_level_month]]),
        ),row=1, col=2
)

fig.add_trace(
    go.Table(
        header=dict(
            values=list(['Daily', '14 days', '30 days'])
        ),
        cells=dict(values=[[yc_slope_d],
                           [yc_slope_biweek],
                           [yc_slope_month]]),
        ),row=2, col=2
)

fig.add_trace(
    go.Table(
        header=dict(
            values=list(['Daily', '14 days', '30 days'])
        ),
        cells=dict(values=[[yc_curve_d],
                           [yc_curve_biweek],
                           [yc_curve_month]]),
        ),row=3, col=2
)


# %%

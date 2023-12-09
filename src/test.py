import plotly.graph_objects as go
import numpy as np


Y = list(range(0, 700, 100))
C = list(range(-50, 125, 25))


trace_TR = go.Scatter(x=Y, y=C, mode='lines+markers', name='S = f(Y)', line=dict(color='black'))
# trace_MR = go.Scatter(x=Qd, y=P, mode='lines+markers', name='Demand Curve', line=dict(color='dimgray'))
# trace_TC = go.Scatter(x=Q, y=TC, mode='lines', name='Total Cost (TC)', line=dict(color='darkgray'))
# trace_PROFIT = go.Scatter(x=Q, y=PROFIT, mode='lines', name='Profit', line=dict(color='gray'))

layout = go.Layout(
    title="Saving Curve",
    xaxis=dict(title='Income (Y)', range=(0, 700)),
    yaxis=dict(title='Savings (S)', range=(-50, 125)),
    showlegend=True,
    autosize=False,
    width=1080,
    height=1080,
)

fig = go.Figure(data=[trace_TR], layout=layout)
fig.add_shape(type='line', x0=0, y0=50, x1=max(Y)+100, y1=50, line=dict(color='dimgray', width=2), name='Investment (I)')
fig.add_shape(type='line', x0=0, y0=0, x1=max(Y)+100, y1=0, line=dict(color='black', width=2), name='X-axis')
fig.add_shape(type='line', x0=0, y0=0, x1=0, y1=max(C)+25, line=dict(color='black', width=2), name='Y-axis')

for x, y in zip(Y, C):
    vertical_line = dict(type='line', x0=x, x1=x, y0=0, y1=y, line=dict(dash='dash'))
    horizontal_line = dict(type='line', x0=0, x1=x, y0=y, y1=y, line=dict(dash='dash'))
    fig.add_shape(vertical_line)
    fig.add_shape(horizontal_line)

# fig.add_shape(type='line', x0=15, y0=0, x1=15, y1=2, line=dict(color='red', width=2), name='X-axis')
# fig.add_shape(type='line', x0=0, y0=2, x1=15, y1=2, line=dict(color='red', width=2), name='Y-axis')

fig.write_image("Plots/SavingsWithConstantInvestmentCurve.png")
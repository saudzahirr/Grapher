import plotly.graph_objects as go
import numpy as np



def Plot(X, Y, C=None, mode="markers+lines",
         name="", title="", x_title="", y_title="", line_title="I (Investment)",
         x_range=None, y_range=None,
         show_grid=False, add_45_deg_line=False, add_line=False, add_line_show_grid=False):

    if x_range is None and y_range is None:
        x_range=(min(X), max(X) + 0.5)
        y_range=(min(Y), max(Y) + 0.5)

    trace = go.Scatter(x=X, y=Y, mode=mode, name=name, marker=dict(color='black', size=8), line=dict(color='black'))
    layout = go.Layout(
        title=title,
        xaxis=dict(title=x_title, linecolor='black', linewidth=2, range=x_range),
        yaxis=dict(title=y_title, linecolor='black', linewidth=2, range=y_range),
        showlegend=True,
        autosize=False,
        width=1440,
        height=1440,
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.add_shape(type='line', x0=-max(X), y0=0, x1=max(X), y1=0, line=dict(color='black', width=2), name='X-axis')
    fig.add_shape(type='line', x0=0, y0=-max(Y), x1=0, y1=max(Y), line=dict(color='black', width=2), name='Y-axis')

    if add_45_deg_line:
        fig.add_trace(go.Scatter(x=X, y=X, mode="lines", name=x_title, marker=dict(color='black', size=8), line=dict(color='rgb(128, 128, 128)')))

    if add_line:
        fig.add_trace(go.Scatter(x=X, y=C, mode="lines", name=line_title, marker=dict(color='black', size=8), line=dict(color='rgb(128, 128, 128)')))

    fig.add_trace(trace)

    if show_grid:
        for x, y in zip(X, Y):
            vertical_line = dict(type='line', x0=x, x1=x, y0=0, y1=y, line=dict(dash='dash'))
            horizontal_line = dict(type='line', x0=0, x1=x, y0=y, y1=y, line=dict(dash='dash'))
            fig.add_shape(vertical_line)
            fig.add_shape(horizontal_line)
        
        if add_line_show_grid and add_line:
            for x, y in zip(X, C):
                vertical_line = dict(type='line', x0=x, x1=x, y0=0, y1=y, line=dict(dash='dash'))
                horizontal_line = dict(type='line', x0=0, x1=x, y0=y, y1=y, line=dict(dash='dash'))
                fig.add_shape(vertical_line)
                fig.add_shape(horizontal_line)

    fig.write_image(f"{'_'.join(title.split())}.png", **{"width": 1080, "height": 1080})


def Linear():
    x = np.linspace(-5, 5, 20)
    y = 1.5*x - 2
    Plot(x, y,
         title="$y = { {3 \\over 2} x} - 2$",
         x_title="x", y_title="f(x)",
         x_range=(min(x), max(x)),
         y_range=(min(y), max(y)),
         show_grid=True)


def Quadratic():
    x = np.linspace(-5, 5, 20)
    y = 3*x**2 - 2*x - 7
    Plot(x, y,
         title="$y = 3x^2 + x - 7$",
         x_title="x", y_title="f(x)",
         x_range=(min(x), max(x)),
         y_range=(min(y), max(y)),
         show_grid=True)


def Cubic():
    x = np.linspace(-5, 5, 20)
    y = x**3 + 4*x - 1
    Plot(x, y,
         title="$y = x^3 + 4x - 1$",
         x_title="x", y_title="f(x)",
         x_range=(min(x), max(x)),
         y_range=(min(y), max(y)),
         show_grid=True)


def IncomeAndAggregateDemand():
    income = list(range(0, 500, 100))
    aggregate_demand = list(range(60, 500, 80))
    Plot(X=income, Y=aggregate_demand,
         title="$AD = f(Y)$",
         x_title="Income (Y)", y_title="Aggregate Demand (AD)",
         show_grid=True, add_45_deg_line=True)


def IncomeAndSavings():
    income = list(range(0, 600, 100))
    savings = list(range(-50, 100, 25))
    Plot(income, savings,
         title="$S = f(Y)$",
         x_title="Income (Y)", y_title="Savings (S)",
         show_grid=True)


def IncomeAndSavingsWithConstantInvestment():
    income = list(range(0, 700, 100))
    savings = list(range(-50, 120, 25))
    Plot(income, savings, [50]*len(income),
         title="$S = f(Y)$",
         x_title="Income (Y)", y_title="Savings (S)",
         show_grid=True,
         add_line=True)


def IncomeAndInvestment():
    income = list(range(0, 600, 100))
    investment = list(range(40, 100, 10))
    Plot(income, investment,
         title="$I = f(Y)$",
         x_range=(0, max(income)), y_range=(0, max(investment)),
         x_title="Income (Y)", y_title="Investment (I)",
         show_grid=True)


def IncomeAndConsumption():
    income = list(range(0, 350, 50))
    consumption = list(range(20, 300, 40))
    Plot(income, consumption,
         title="$C = f(Y)$",
         x_title="Income (Y)", y_title="Consumption (C)",
         show_grid=True,
         add_45_deg_line=True)


def QuantityAndTotalRevenue():
    def TR(Q):
        return 20*Q - 2*Q**2

    quantity = np.linspace(1, 6, 6)
    total_revenue = TR(quantity)
    Plot(quantity, total_revenue, mode="lines",
         title="$TR = 20Q - 2Q^{2}$",
         x_range=(0, max(quantity) + 1), y_range=(0, max(total_revenue) + 1),
         x_title="Quantity (Q)", y_title="Total Revenue (TR)",
         show_grid=True)


def QuantityAndMarginalRevenue():
    def MR(Q):
        return 20 - 4*Q

    quantity = np.linspace(1, 4, 4)
    marginal_revenue = MR(quantity)
    Plot(quantity, marginal_revenue, mode="lines",
         title="$MR = 20 - 4Q$",
         x_range=(0, max(quantity) + 1), y_range=(0, max(marginal_revenue) + 1),
         x_title="Quantity (Q)", y_title="Marginal Revenue (MR)",
         show_grid=True)


def QuantityAndAveragelRevenue():
    def AR(Q):
        return 20 - 2*Q

    quantity = np.linspace(1, 9, 10)
    average_revenue = AR(quantity)
    Plot(quantity, average_revenue, mode="lines",
         title="$AR = 20 - 2Q$",
         x_range=(0, max(quantity) + 1), y_range=(0, max(average_revenue) + 1),
         x_title="Quantity (Q)", y_title="Average Revenue (AR)",
         show_grid=True)


def PriceAndQuantityDemand():
    def Q(P):
        return 10 - 2*P

    price = np.arange(1, 5, 1)
    quantity = Q(price)
    Plot(price, quantity,
         title="$Q_d = f(P)$",
         x_range=(0, max(price) + 1), y_range=(0, max(quantity) + 1),
         x_title="Price (P)", y_title="Quantity Demand (Qd)",
         show_grid=True)


def PriceAndQuantitySupply():
    P = np.arange(1, 6, 1)
    Qs = np.arange(10, 30, 4)
    Plot(P, Qs,
         title="$Q_s = f(P)$",
         x_range=(0, max(P) + 1), y_range=(0, max(Qs) + 1),
         x_title="Price (P)", y_title="Quantity Supply (Qs)",
         show_grid=True)


def PQdQs():
    Qs = [13, 15, 17, 19]
    Qd = [20, 15, 10, 5]
    P = [1, 2, 3, 4]
    Plot(P, Qd, Qs,
         title="$Q_d = f(P) \\\\ Q_s = f(P)$",
         x_range=(0, max(P) + 1), y_range=(0, max(Qd) + 1),
         x_title="Price (P)", y_title="Quantity (Qd/Qs)",
         line_title="Qs = f(P)",
         show_grid=True, add_line=True, add_line_show_grid=True)


def ComparisonPlot():
    Q = np.linspace(0, 10, 100)

    TR = 24 * Q - 1.5 * Q**2
    MR = 24 - 3 * Q
    TC = 8 + 4 * Q + 0.5 * Q**2
    PROFIT = -8 + 20 * Q - 2 * Q**2

    C = [0.937742, 0.417424, 9.58258, 2, 23/4 - np.sqrt(273)/4, 23/4 + np.sqrt(273)/4, 1.24041, 5.15959]
    V = [24 - 3 * C[0], 24*C[1] - 1.5*C[1]**2, 24*C[2] - 1.5*C[2]**2, 24 - 3*C[3], 24 - 3*C[4], 24 - 3*C[5], 8 + 4*C[6] + 0.5*C[6]**2, 8 + 4*C[7] + 0.5*C[7]**2]

    trace_TR = go.Scatter(x=Q, y=TR, mode='lines', name='Total Revenue (TR)', line=dict(color='black'))
    trace_MR = go.Scatter(x=Q, y=MR, mode='lines', name='Marginal Revenue (MR)', line=dict(color='dimgray'))
    trace_TC = go.Scatter(x=Q, y=TC, mode='lines', name='Total Cost (TC)', line=dict(color='darkgray'))
    trace_PROFIT = go.Scatter(x=Q, y=PROFIT, mode='lines', name='Profit', line=dict(color='gray'))

    layout = go.Layout(
        title='$TR(Q) = 24Q - 1.5Q^2, \\\\ MR(Q) = 24 - 3Q, \\\\ TC(Q) = 8 + 4Q + 0.5Q^2, \\\\ \Pi(Q) = -8 + 20Q - 2Q^2$',
        xaxis=dict(title='Quantity (Q)'),
        yaxis=dict(title='Revenue, Cost and Profit'),
        showlegend=True,
        autosize=False,
        width=1080,
        height=1080,
    )

    fig = go.Figure(data=[trace_TR, trace_MR, trace_TC, trace_PROFIT], layout=layout)
    fig.add_shape(type='line', x0=min(Q), y0=0, x1=max(Q), y1=0, line=dict(color='black', width=2), name='X-axis')
    fig.add_shape(type='line', x0=0, y0=min(TR), x1=0, y1=max(TR) + 10, line=dict(color='black', width=2), name='Y-axis')

    for c, v in zip(C, V):
        intersection_line = dict(type='line', x0=c, x1=c, y0=0, y1=v, line=dict(dash='dash'))
        fig.add_shape(intersection_line)

    fig.write_image("Comparison_Plot.png")


if __name__ == "__main__":
    Linear()
    Quadratic()
    Cubic()
    IncomeAndAggregateDemand()
    IncomeAndSavings()
    IncomeAndSavingsWithConstantInvestment()
    IncomeAndConsumption()
    IncomeAndInvestment()
    QuantityAndTotalRevenue()
    QuantityAndMarginalRevenue()
    QuantityAndAveragelRevenue()
    PriceAndQuantityDemand()
    PriceAndQuantitySupply()
    PQdQs()
    ComparisonPlot()

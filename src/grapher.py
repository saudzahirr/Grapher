import plotly.graph_objects as go
import numpy as np
from typing import List, Callable

class Grapher:
    def __init__(self, X: List, Y: List[List] | List[Callable], make_smooth: bool = False, **kwargs):
        """
        Initialize the Grapher object.

        Arguments:
            X (List): List of x-axis values.
            Y (List[List] | List[Callable]): List of y-axis values as a list of lists or a list of callable functions.
            make_smooth (bool): If True, smooth the x-axis values.
            **kwargs: Additional keyword arguments.
        """
        self.X = X
        self.Y = Y if isinstance(Y[0], list) else self._generate_y_data(Y)
        self.display = kwargs.get("display", False)

        if make_smooth:
            self._smooth()

        self.traces = self.get_traces()

    def _generate_y_data(self, callables: List[Callable]) -> List[List]:
        """
        Generate y-axis data from callable functions.

        Arguments:
            callables (List[Callable]): List of callable functions.

        Returns:
            List[List]: List of y-axis values for each callable.
        """
        Y = []
        for function in callables:
            y = []
            for x in self.X:
                y.append(function(x))
            Y.append(y)
        return Y

    def _smooth(self):
        """
        Smooth the x-axis values.
        """
        step_size = (max(self.X) - min(self.X)) / 100
        self.X = list(np.arange(min(self.X), max(self.X), step_size))

    def get_traces(self):
        """
        Generate plotly traces for each y-axis data.

        Returns:
            List[go.Scatter]: List of plotly traces.
        """
        traces = []
        for idx, y in enumerate(self.Y):
            trace = go.Scatter(x=self.X, y=y,
                               mode="lines+markers",
                               marker=dict(color=f"rgb({idx * 30}, {idx * 50}, {idx * 70})", size=8),
                               line=dict(color=f"rgb({idx * 30}, {idx * 50}, {idx * 70})"))
            traces.append(trace)
        return traces

    def _add_grid_lines(self, fig):
        """
        Add grid lines to the plot.

        Arguments:
            fig: Plotly figure object.
        """
        for x, y_values in zip(self.X, zip(*self.Y)):
            for y in y_values:
                vertical_line = dict(type="line", x0=x, x1=x, y0=0, y1=y, line=dict(dash="dash"))
                fig.add_shape(vertical_line)

    def _add_45_degree_line(self, fig):
        """
        Add a 45-degree line to the plot.

        Arguments:
            fig: Plotly figure object.
        """
        fig.add_trace(go.Scatter(x=self.X, y=self.X, mode="lines",
                                 name="45-degree line", marker=dict(color="black", size=8),
                                 line=dict(color="rgb(128, 128, 128)")))

    def plot(self, title="", x_title="", y_title="", x_range=None, y_range=None,
             add_45_deg_line=False, add_line=False, line_title="", show_grid=False,
             add_line_show_grid=False):
        """
        Plot the graph.

        Arguments:
            title (str): Title of the plot.
            x_title (str): Title of the x-axis.
            y_title (str): Title of the y-axis.
            x_range (Tuple): Range of values for the x-axis.
            y_range (Tuple): Range of values for the y-axis.
            add_45_deg_line (bool): If True, add a 45-degree line to the plot.
            add_line (bool): If True, add a custom line to the plot.
            line_title (str): Title of the custom line.
            show_grid (bool): If True, display grid lines.
            add_line_show_grid (bool): If True, add a line and show grid lines.
        """
        if x_range is None and y_range is None:
            x_range = (min(self.X), max(self.X) + 0.5)
            y_range = (min(min(y) for y in self.Y), max(max(y) for y in self.Y) + 0.5)

        layout = go.Layout(
            title=title,
            xaxis=dict(title=x_title, linecolor="black", linewidth=2, range=x_range),
            yaxis=dict(title=y_title, linecolor="black", linewidth=2, range=y_range),
            showlegend=True,
            autosize=False,
            width=1440,
            height=1440,
        )

        fig = go.Figure(data=self.traces, layout=layout)

        fig.add_shape(type="line", x0=-max(self.X), y0=0, x1=max(self.X), y1=0,
                      line=dict(color="black", width=2), name="X-axis")
        fig.add_shape(type="line", x0=0, y0=-max(min(y) for y in self.Y), x1=0, y1=max(max(y) for y in self.Y),
                      line=dict(color="black", width=2), name="Y-axis")

        if add_45_deg_line:
            self._add_45_degree_line(fig)

        if add_line:
            C = [0] * len(self.X)
            fig.add_trace(go.Scatter(x=self.X, y=C, mode="lines",
                                     name=line_title, marker=dict(color="black", size=8),
                                     line=dict(color="rgb(128, 128, 128)")))

        if show_grid:
            self._add_grid_lines(fig)

        fig.write_image(f"{'_'.join(title.split())}.png", **{"width": 1080, "height": 1080})
        if self.display:
            fig.show()


if __name__ == "__main__":
    x_values = np.arange(0, 2, 1/10)
    functions = [lambda x: x**2, lambda x: 2 * x + 1]

    graph = Grapher(X=x_values, Y=functions, make_smooth=False)
    graph.plot(title="Example", x_title="X-axis", y_title="Y-axis", add_45_deg_line=True, add_line=True,
               line_title="Custom Line", show_grid=True)

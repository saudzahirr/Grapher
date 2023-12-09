import plotly.graph_objects as go
import numpy as np
from typing import List, Callable

class Grapher:
    def __init__(self, X: List, Y: List[List] | List[Callable], make_smooth: bool()):
        self.X = X
        self.Y = Y if isinstance(Y[0], list) else self._generate_y_data(Y)

        if make_smooth:
            self._smooth()

    def _generate_y_data(self, callables: List[Callable]) -> List[List]:
        Y = []
        for function in callables:
            y = []
            for x in self.X:
                y.append(function(x))
            Y.append(y)
        return Y
    
    def _smooth(self):
        step_size = (max(self.X) - min(self.X))/100
        self.X = list(np.arange(min(self.X), max(self.X), step_size))
    
    def plot(self):
        pass
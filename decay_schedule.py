class LinearDecay:
    def __init__(self, start: float, stop: float, steps: int) -> None:
        self.start = start
        self.stop = stop
        self.steps = steps
        self.decay = (self.stop - self.start) / self.steps
    
    def get(self, step: int) -> float:
        decayed_val = self.start + self.decay * step
        return max(decayed_val, self.stop)


class PowerDecay:
    def __init__(self, start: float, stop: float, factor: float) -> None:
        self.start = start
        self.stop = stop
        self.factor = factor
    
    def get(self, step: int) -> float:
        decayed_val = self.start * self.factor**step
        return max(decayed_val, self.stop)

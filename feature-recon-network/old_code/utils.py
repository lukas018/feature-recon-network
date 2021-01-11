class Averager():

    def __init__(self, alpha):
        self.alpha = alpha
        self.v = None
        self.count

    def _update(self, v):
        if self.count > 0:
            self.v = (self.alpha * v) + ((1. *self.alpha) * self.v)
        else:
            self.v = v

    def __call__(self, v=None):
        if v is not None:
            self._update(v)
        return self.v


class Statistics():

    def __init__(self, writer, alpha=0.95):
        self.writer = writer
        self.averagers = defaultdict(partial(Averager, alpha=alpha))

    def __setitem__(self, key, value):
        self.averagres[key](value)

    def write(self, step):
        for key, averager in self.averages.items():
            writer.add_scalar(key, averager(), step)

    def __str__(self):
        return '\n'.join([f"{key}:{averager()}"
                          for key, averager
                          in self.averages.items()])


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


class Stats():

    def __init__(self, moving_avg_window=None):
        self._values = []
        self.moving_avg_window = moving_avg_window

    @property
    def array(self):
        return np.array(self._values)

    @property
    def mean(self):
        if self.moving_avg_window is not None:
            return np.mean(self.array[-self.moving_avg_window:])
        else:
            return np.mean(self.array)

    @property
    def std(self):
        return np.std(self.array)

    @property
    def c95(self):
        return self.conf_int(0.95)

    def conf_int(self, c):
        a = self.array
        interval = st.t.interval(c, len(a)-1, loc=np.mean(a), scale=st.sem(a))
        return interval

    def __call__(self, *args):
        self._values.extend([*args])

class SummaryGroup():

    def __init__(self):
        self.stats = defaultdict(Statistics)

    @classmethod
    def from_dicts(cls, metrics):
        metrics = {key: [v[key] for v in metrics] for key in metrics[0].keys()}
        sg = SummaryGroup()
        for key, vs in metrics.items():
            sg(key, *vs)
        return sg

    def __setitem__(self, key, *values):
        self.stats[key](value)

    def write(self, writer, step, prefix=None):

        for key, stats in self.stats.items():
            key = key if prefix is None else f"{prefix}/{key}"
            writer.add_scalar(key, stats.mean, step)

    def __str__(self):
        strs = [f"{key}: {stat.mean:.3f} ± {stat.std:.3f}"
                          for key, stat
                          in self.stats.items()]
        return '\n'.join(strs)

    def join(sg):
        new_sg = SummaryGroup(self.writer)
        new_sg.stats = copy.deepcopy(new_sg.stats)
        new_sg.stats.update(sg.stats)
        return new_sg

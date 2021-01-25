from .meta_baseline import MetaBaseline


class ProtypicalNetwork(MetaBaseline):
    """Simple implementation of prototypical network:
    https://papers.nips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf

    THis approach shares many similarities to other metric-learning based
    approaches.  However, unlike e.g. new Meta-Baseline or FRN this method
    doesn't utilize any pretraining.
    """

    def __init__(self, model):
        super().__init__(model)
        self.temperature = 1.0

        def dist_fn(features, centroids):
            return -((features - centroids) ** 2).sum(dim=2)

        self.dist_fn = dist_fn

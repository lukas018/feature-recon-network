
class ProtypicalNetwork(Metabaseline):
    """Implementation of prototypical network:
    https://papers.nips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf

    """

    def __init__(self, model):
        super(self).__init__(model)
        self.temperature = 1.

        def dist_fn(features, centroids):
            return -((features - centriods)**2).sum(dim=2)

        self.dist_fn = dist_fn

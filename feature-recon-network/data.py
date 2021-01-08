import learn2learn.data.transforms

class RandomNWays(CythonRandomNWays):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)
    **Description**
    Keeps samples from N random labels present in the task description.
    **Arguments**
    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **n** (int, *optional*, default=2) - Number of labels to sample from the task
        description's labels.
    """

    def __init__(self, dataset, nrange=2):
        super(NWays, self).__init__(dataset=dataset, nrange=nrange)


cdef class CythonRandomNWays(TaskTransform):

    cdef public:
        int nmin
        int nmax
        dict indices_to_labels

    def __init__(self, dataset, nrange=(4,5)):
        super(CythonNWays, self).__init__(dataset)
        self.nmin = nrange[0]
        self.nmax = nrange[1]
        self.indices_to_labels = dict(dataset.indices_to_labels)

    def __reduce__(self):
        return CythonNWays, (self.dataset, (self.nmin, self.nmax))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef new_task(self):  # Efficient initializer
        cdef list labels = self.dataset.labels
        cdef list task_description = []
        cdef dict labels_to_indices = dict(self.dataset.labels_to_indices)
        n = random.randint(self.nmin, self.nmax)
        classes = random.sample(labels, k=n)
        for cl in classes:
            for idx in labels_to_indices[cl]:
                task_description.append(DataDescription(idx))
        return task_description

    def __call__(self, list task_description):
        if task_description is None:
            return self.new_task()
        cdef list classes = []
        cdef list result = []
        cdef set set_classes = set()
        cdef DataDescription dd
        for dd in task_description:
            set_classes.add(self.indices_to_labels[dd.index])
        classes = <list>set_classes
        n = random.randint(self.nmin, self.nmax)
        classes = random.sample(labels, k=n)
        for dd in task_description:
            if self.indices_to_labels[dd.index] in classes:
                result.append(dd)
        return result


class RandomKShots(CythonKShots):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)
    **Description**
    Keeps K samples for each present labels.
    **Arguments**
    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **k** (int, *optional*, default=1) - The number of samples per label.
    * **replacement** (bool, *optional*, default=False) - Whether to sample with replacement.
    """

    def __init__(self, dataset, krange=1, replacement=False):
        super(KShots, self).__init__(dataset=dataset,
                                     krange=krange,
                                     replacement=replacement)


cdef class CythonRandomKShots(TaskTransform):

    cdef public:
        long kmin
        long kmax
        bool replacement

    def __init__(self, dataset, krange=(11, 15), replacement=False):
        super(CythonKShots, self).__init__(dataset)
        self.dataset = dataset
        self.kmin = krange[0]
        self.kmax = krange[1]
        self.replacement = replacement

    def __reduce__(self):
        return CythonKShots, (self.dataset, (self.kmin, self.kmax), self.replacement)

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()

        class_to_data = collections.defaultdict(list)
        for dd in task_description:
            cls = self.dataset.indices_to_labels[dd.index]
            class_to_data[cls].append(dd)
        if self.replacement:
            def sampler(x, k):
                return [copy.deepcopy(dd)
                        for dd in random.choices(x, k=k)]
        else:
            sampler = random.sample

        # Sample a suitable k
        k = random.randint(self.kmin, self.kmax)
        return list(itertools.chain(*[sampler(dds, k=k) for dds in class_to_data.values()]))

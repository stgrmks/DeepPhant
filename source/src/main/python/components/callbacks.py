_author__ = 'MSteger'

class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.hist = []

    def update(self, val, n = 1):
        self.hist.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def best(self, min = True):
        return max(self.hist) if not min else min(self.hist)
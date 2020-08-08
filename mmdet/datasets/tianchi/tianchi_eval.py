import time
from collections import defaultdict

import numpy as np


class TianchiEval:
    """
    Usage:
        E = TianchiEval(tianchiGt, tianchiDt)
        E.evaluate()
        E.accumulate()
        E.summarize()
    """

    def __init__(self, tianchiGt=None, tianchiDt=None):
        self.tianchiGt = tianchiGt
        self.tianchiDt = tianchiDt
        # per-study per-part evaluation results
        self.evalStudies = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        # parameters
        self.params = None
        self._paramsEval = {}  # parameters for evaluation
        self.stats = {}  # results summarization

        if tianchiGt is not None:
            self.params.study_ids = sorted(tianchiGt.get_study_ids())
            self.params.cat_ids = sorted(tianchiGt.get_cat_ids())

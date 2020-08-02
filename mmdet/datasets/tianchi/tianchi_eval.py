import time
from collections import defaultdict
from pycocotools.cocoeval import COCOeval


class TianchiEval(COCOeval):

    def _prepare(self):
        """Prepare ._gts and ._dts for evaluation based on params."""
        p = self.params

        if p.useCats:
            gts = self.tianchiGt.load_anns(self.tianchiGt.get_ann_ids())
            dts = self.tianchiDt.load_anns(self.tianchiDt.get_ann_ids())
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['study_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['study_id'], dt['category_id']].append(dt)
        self.eval_stuies = defaultdict(list)
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        """Run per image evaluation on given images and store
        results (a list of dict) in self.eval_stuies
        """
        tic = time.time()
        print('Running per image evaluation...')

        self._prepare()

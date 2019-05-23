"""Kinetics Human Action Dataset"""
from __future__ import absolute_import

import os
import csv
from collections import namedtuple
from ... import ndarray as nd
from ..._ffi.ndarray import DECORDContext

class KineticsHumanActionDataset(object):
    def __init__(self, root, ctx=nd.cpu(0), splits=('kinetics_600_val.csv',), lazy=True):
        self._root = os.path.abspath(os.path.expanduser(root))
        self._splits = splits
        if isinstance(ctx, DECORDContext):
            ctx = [ctx]
        self._ctx = ctx
        self._lazy = lazy

        # initialize annotations
        self._load_annotations()
        
    def _load_annotations(self):
        self._items = []
        for split in self._splits:
            items = self._read_csv(split)
            self._items += items

    def _read_csv(self, split):
        items = []
        fn = os.path.join(self._root, split)
        with open(fn, newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            Row = namedtuple('Row', headers)
            for r in reader:
                row = Row(*r)
                items.append({'label': row.label,
                              'youtube_id': row.youtube_id,
                              'time_start': row.time_start,
                              'time_end': row.time_end,
                              'split': row.split})
        return items

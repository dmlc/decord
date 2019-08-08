"""Kinetics Human Action Dataset"""
from __future__ import absolute_import

import os
import csv
from collections import namedtuple
import numpy as np
from ... import ndarray as nd
from ..._ffi.ndarray import DECORDContext
from ...video_reader import VideoReader

class KineticsHumanActionDataset(object):
    """
    Kinetics Human Action Dataset. Powered by internal VideoReader.

    Parameters
    ----------
    root : str
        Root path of the kinetics dataset.
    num_samples : int, default is 8
        Number of frames to be sampled from each video clip.
    width : int, default is 320
        Frame resize width.
    height : int, default is 240
        Frame resize height.
    ctx : DECORDContext, default is cpu(0)
        The video decoding and returned array context. Can be a list of ctx.
    splits : tuple, default is the val split
        The splits indicate either train/val/test splits will be loaded. You can combine multiple sets.

    """
    def __init__(self, root, num_samples=8, width=320, height=240, ctx=nd.cpu(0), splits=('kinetics_600_val.csv',)):
        self._root = os.path.abspath(os.path.expanduser(root))
        assert num_samples > 0
        self._num_samples = num_samples
        assert width > 0
        self._width = width
        assert height > 0
        self._height = height
        self._splits = splits
        if isinstance(ctx, DECORDContext):
            ctx = [ctx]
        assert isinstance(ctx, (tuple, list))
        self._ctx = ctx

        # initialize annotations
        self._items = []
        self._load_annotations()

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        ctx = self._ctx[idx % len(self._ctx)]
        item = self._items[idx]
        label = item['label']
        yid = item['youtube_id']
        ts = int(item['time_start'])
        te = int(item['time_end'])
        filename = os.path.join(self._root, label, yid + '_{:06d}_{:06d}.mp4'.format(ts, te))
        assert os.path.isfile(filename), '{} does not exist'.format(filename)
        vr = VideoReader(filename, ctx=ctx, width=self._width, height=self._height)
        total_frames = len(vr)
        assert total_frames > self._num_samples, (
            "Sampling {} out of {} frames is not possible".format(self._num_samples, total_frames))
        sample_indices = [int(np.round(x)) for x in np.linspace(0, total_frames-1, self._num_samples)]
        arrs = vr.get_batch(sample_indices)
        del vr
        return arrs

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

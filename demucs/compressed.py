# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from concurrent import futures

import musdb
import torch
from .audio import AudioFile

import yaml
from .inference import Inferencer
from argparse import Namespace


def get_musdb_tracks(root, *args, **kwargs):
    mus = musdb.DB(root, *args, **kwargs)
    return {track.name: track.path for track in mus}


class StemsSet:
    def __init__(self, tracks, metadata, duration=None, stride=1, samplerate=44100, channels=2, speaker_emb=False):

        self.metadata = []
        for name, path in tracks.items():
            meta = dict(metadata[name])
            meta["path"] = path
            meta["name"] = name
            self.metadata.append(meta)
            if duration is not None and meta["duration"] < duration:
                raise ValueError(f"Track {name} duration is too small {meta['duration']}")
        self.metadata.sort(key=lambda x: x["name"])
        self.duration = duration
        self.stride = stride
        self.channels = channels
        self.samplerate = samplerate
        self.use_speaker_emb = speaker_emb
        self.args = Namespace(attr='demucs/attr.pkl', config='demucs/config.yaml', model='demucs/vctk_model.ckpt')
        with open(self.args.config) as f:
            self.config = yaml.load(f)
        self.inferencer = Inferencer(config=self.config, args=self.args)

    def __len__(self):
        return sum(self._examples_count(m) for m in self.metadata)

    def _examples_count(self, meta):
        if self.duration is None:
            return 1
        else:
            return int((meta["duration"] - self.duration) // self.stride + 1)

    def track_metadata(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            return meta

    def __getitem__(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            streams = AudioFile(meta["path"]).read(seek_time=index * self.stride,
                                                   duration=self.duration,
                                                   channels=self.channels,
                                                   samplerate=self.samplerate) # size is 5 stems * 2 channels * T
            #print(meta["path"])
            #print(self.samplerate)
            if self.use_speaker_emb:
                embedding = self.inferencer.infer_speaker(streams[0], samplerate=self.samplerate) # give stream of mixture only
                embedding = torch.unsqueeze(embedding, 1)
            else:
                embedding = self.inferencer.infer_content(streams[0], samplerate=self.samplerate) # give stream of mixture only
            #print("compressed streams shape:", streams.shape)
            #print("compressed embedding shape:", embedding.shape)
            #print()
            return (streams - meta["mean"]) / meta["std"], embedding


def _get_track_metadata(path):
    # use mono at 44kHz as reference. For any other settings data won't be perfectly
    # normalized but it should be good enough.
    audio = AudioFile(path)
    mix = audio.read(streams=0, channels=1, samplerate=44100)
    return {"duration": audio.duration, "std": mix.std().item(), "mean": mix.mean().item()}


def build_metadata(tracks, workers=10):
    pendings = []
    with futures.ProcessPoolExecutor(workers) as pool:
        for name, path in tracks.items():
            pendings.append((name, pool.submit(_get_track_metadata, path)))
    return {name: p.result() for name, p in pendings}


def build_musdb_metadata(path, musdb, workers):
    tracks = get_musdb_tracks(musdb)
    metadata = build_metadata(tracks)
    path.parent.mkdir(exist_ok=True, parents=True)
    json.dump(metadata, open(path, "w"))

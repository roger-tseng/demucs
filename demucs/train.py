# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import apply_model, average_metric, center_trim

def collate_tuple(batch):
    '''
    Each batch consists of (streams, embeddings)
    '''
    streams = torch.stack(tuple(i[0] for i in batch))
    #print("streams size is:", streams.size())
    embeddings = torch.stack(tuple(i[1] for i in batch))
    #print("emb size is:", embeddings.size())
    #print("done collating")
    return [streams, embeddings]

def train_model(epoch,
                dataset,
                model,
                criterion,
                optimizer,
                augment,
                repeat=1,
                device="cpu",
                seed=None,
                workers=4,
                world_size=1,
                batch_size=16):

    if world_size > 1:
        sampler = DistributedSampler(dataset)
        sampler_epoch = epoch * repeat
        if seed is not None:
            sampler_epoch += seed * 1000
        sampler.set_epoch(sampler_epoch)
        batch_size //= world_size
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, collate_fn=collate_tuple)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True, collate_fn=collate_tuple)
    current_loss = 0
    for repetition in range(repeat):
        tq = tqdm.tqdm(loader,
                       ncols=120,
                       desc=f"[{epoch:03d}] train ({repetition + 1}/{repeat})",
                       leave=False,
                       file=sys.stdout,
                       unit=" batch")
        total_loss = 0
        for idx, streams in enumerate(tq):
            if len(streams[0]) < batch_size:
                # skip uncomplete batch for augment.Remix to work properly
                continue
            content_embeddings = streams[1].to(device)
            streams = streams[0].to(device)
            sources = streams[:, 1:]
            #print("idx:", idx, "sources size (train):", sources.size())
            sources = augment(sources)
            #print("idx:", idx, "sources size after augment (train):", sources.size())
            mix = sources.sum(dim=1)
            #print("idx:", idx, "mix size after augment (train):", mix.size())
            #print("idx:", idx, "emb size (train)", content_embeddings.size())
            estimates = model(mix, content_embeddings)
            sources = center_trim(sources, estimates)
            loss = criterion(estimates, sources)
            #print("idx:", idx, "loss:", loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            current_loss = total_loss / (1 + idx)
            tq.set_postfix(loss=f"{current_loss:.4f}")

            # free some space before next round
            del content_embeddings, streams, sources, mix, estimates, loss

        if world_size > 1:
            sampler.epoch += 1

    if world_size > 1:
        current_loss = average_metric(current_loss)
    return current_loss


def validate_model(epoch,
                   dataset,
                   model,
                   criterion,
                   device="cpu",
                   rank=0,
                   world_size=1,
                   shifts=0,
                   split=False):
    indexes = range(rank, len(dataset), world_size)
    tq = tqdm.tqdm(indexes,
                   ncols=120,
                   desc=f"[{epoch:03d}] valid",
                   leave=False,
                   file=sys.stdout,
                   unit=" track")
    current_loss = 0
    for index in tq:
        
        streams, emb = dataset[index]
        print("streams type:", type(streams), "size:", streams.size())
        emb = emb[None, :, :].to(device) # reshape emb to 3D tensor (batch of 1)
        print("emb type:", type(emb), "size:", emb.size())
        # first five minutes to avoid OOM on --upsample models
        streams = streams[..., :15_000_000]
        streams = streams.to(device)
        sources = streams[1:]
        print("idx:", index, "sources size (valid):", sources.size())
        mix = streams[0]
        print("idx:", index, "mix size (valid):", mix.size())
        estimates = apply_model(model, emb, mix, shifts=shifts, split=split)
        loss = criterion(estimates, sources)
        print("idx:", index, "loss (valid):", loss)
        current_loss += loss.item() / len(indexes)
        del estimates, streams, sources

    if world_size > 1:
        current_loss = average_metric(current_loss, len(indexes))
    return current_loss

"""Train GCANet reconstruction and GSF on the complete BEHAVE training split."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def accumulation_divisor(step, batches, accumulate):
    """Preserve the smaller final accumulation window without dropping images."""
    return min(accumulate, batches - (step // accumulate) * accumulate)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data-root', required=True)
    result.add_argument('--dino-checkpoint', required=True)
    result.add_argument('--hand4whole-checkpoint')
    result.add_argument('--output', required=True, help='A new output directory, including when resuming')
    result.add_argument('--gpu', required=True, type=int, help='One physical GPU index')
    result.add_argument('--workers', type=int, default=4)
    result.add_argument('--micro-batch', type=int, default=8)
    result.add_argument('--accumulate', type=int, default=2)
    result.add_argument('--seed', type=int, default=20261014)
    result.add_argument('--resume', metavar='CHECKPOINT', help='Explicit complete-epoch checkpoint')
    return result


def main():
    args = parser().parse_args()
    if args.gpu < 0 or args.workers < 0 or args.micro_batch <= 0 or args.accumulate <= 0:
        raise ValueError('GPU/workers must be nonnegative; batch and accumulation positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['XFORMERS_DISABLED'] = '1'
    import cv2
    import numpy as np
    import torch
    from gcanet.reconstruction.runtime import (initialize, dataset, loader, move,
        seed_all, build_model, OriginalLoss, load_checkpoint, sha256)
    from gcanet.reconstruction.checkpoint import payload, load_model
    from gcanet.reconstruction.throughput import apply_throughput_fixes

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Training requires the one selected GPU')
    initialize(args.data_root, args.dino_checkpoint, args.hand4whole_checkpoint)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    seed_all(args.seed)
    train = dataset('train', training=True)
    test_annotation = Path(args.data_root) / 'base_data/annotations/behave_test.json'
    test_ids = {int(a['id']) for a in json.loads(test_annotation.read_text())['annotations']}
    train_ids = {int(row['ann_id']) for row in train.datalist}
    if train_ids & test_ids:
        raise ValueError('BEHAVE train and test annotation IDs overlap')
    generator = torch.Generator().manual_seed(args.seed)
    batches = loader(train, args.micro_batch, args.workers, True, generator)
    model = build_model(initialize_backbone=not bool(args.resume))
    apply_throughput_fixes(model)
    criterion = OriginalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.5, .999))
    package_root = Path(__file__).resolve().parents[1]
    code_files = sorted((package_root / 'gcanet/reconstruction').rglob('*.py')) + [
        package_root / 'gcanet/gsf.py', package_root / 'gcanet/gsf_supervision.py',
        package_root / 'gcanet/compat.py', Path(__file__).resolve()]
    source_sha256 = {str(p.relative_to(package_root)): sha256(p) for p in code_files}
    config = dict(source_sha256=source_sha256, mode='gsf', epochs=50, micro_batch=args.micro_batch,
        accumulate=args.accumulate, workers=args.workers, seed=args.seed,
        training_protocol='BEHAVE45380_fixed50', final_checkpoint_rule='epoch50',
        reconstruction='GCANetReconstruction', fusion='GeometryConditionedSemanticFusion',
        train_annotation_sha256=sha256(Path(args.data_root) / 'base_data/annotations/behave_train.json'))
    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        for key in ('micro_batch', 'accumulate', 'workers', 'seed', 'train_annotation_sha256', 'source_sha256'):
            if checkpoint['config'].get(key) != config[key]:
                raise ValueError(f'Resume recipe mismatch: {key}')
        if not 0 <= checkpoint['epoch'] < 50:
            raise ValueError('Resume requires a complete epoch before epoch 50')
        load_model(model, checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer'])
        state = checkpoint['rng']
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state_all(state['cuda'])
        generator.set_state(state['loader'])
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    calls = {'tg': 0, 'dino': 0}
    def count_tg(*_): calls['tg'] += 1
    def count_dino(*_): calls['dino'] += 1
    model.tgformer.register_forward_hook(count_tg)
    model.evidence.dino.register_forward_hook(count_dino)
    with (output / 'training.jsonl').open('x', buffering=1) as log:
        for epoch in range(start_epoch, 51):
            rate = 1e-4 if epoch <= 30 else 1e-5
            for group in optimizer.param_groups:
                group['lr'] = rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals, seen, updates = {}, 0, 0
            started = time.monotonic()
            for step, (inputs, targets, meta) in enumerate(batches):
                inputs, targets, meta = move(inputs), move(targets), move(meta)
                calls.update(tg=0, dino=0)
                prediction = model(inputs)
                if calls != {'tg': 1, 'dino': 1}:
                    raise AssertionError(f'GCANet must perform one GSF and TG pass: {calls}')
                loss, parts = criterion(prediction, targets, meta)
                extra, extra_parts = model.evidence_auxiliary_loss(targets, meta)
                loss = loss + extra
                parts.update(extra_parts)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, batch {step}')
                (loss / accumulation_divisor(step, len(batches), args.accumulate)).backward()
                update = (step + 1) % args.accumulate == 0 or step + 1 == len(batches)
                if update:
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise FloatingPointError('Nonfinite reconstruction gradient')
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                count = len(inputs['img'])
                seen += count
                for key, value in {'total': loss, **parts}.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach()) * count
            if seen != 45380:
                raise AssertionError('Incomplete training epoch')
            row = dict(epoch=epoch, examples=seen, updates=updates, lr=rate,
                loss={k: v / seen for k, v in totals.items()}, seconds=time.monotonic() - started)
            log.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            saved = payload(model, epoch, config, optimizer=optimizer.state_dict(),
                rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(), loader=generator.get_state()))
            temporary = output / 'checkpoint.tmp'
            torch.save(saved, temporary)
            temporary.replace(output / ('final.pt' if epoch == 50 else 'latest.pt'))
    (output / 'complete.json').write_text(json.dumps({'epoch': 50, 'training_complete': True,
        'evaluation_performed': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()

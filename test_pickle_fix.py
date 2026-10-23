import multiprocessing


def main():
    import os, pickle, time
    import torch
    from torch.utils.data import DataLoader, RandomSampler

    from libero.libero import get_libero_path
    from libero.lifelong.datasets import SequenceVLDataset, get_dataset

    obs_modality = {
        "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
        "depth": [],
        "low_dim": ["gripper_states", "joint_states"],
    }

    ds_path = os.path.join(
        get_libero_path("datasets"),
        "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
    )
    print("dataset path:", ds_path, os.path.exists(ds_path), flush=True)

    t0 = time.time()
    seq_ds, shape_meta = get_dataset(ds_path, obs_modality, seq_len=10)
    dataset = SequenceVLDataset(seq_ds, torch.zeros(768))
    print(f"dataset built in {time.time()-t0:.1f}s, n_seq={len(dataset)}", flush=True)

    # simulate compute_flops: getitem in main process reopens hdf5 handle
    t0 = time.time()
    _ = dataset[0]
    print(f"one getitem: {time.time()-t0:.3f}s, _hdf5_file open: {seq_ds._hdf5_file is not None}", flush=True)
    try:
        pickle.dumps(dataset)
        print("pickle before close: OK", flush=True)
    except Exception as e:
        print("pickle before close: FAIL ->", e, flush=True)

    # apply the same close as the fix in compute_flops
    seq_ds.close_and_delete_hdf5_handle()
    pickle.dumps(dataset)
    print("pickle after close: OK", flush=True)

    # real spawn-worker dataloader test (with worker_init_fn, as patched in base.py)
    from functools import partial
    from libero.lifelong.datasets import obs_utils_init_worker
    loader = DataLoader(dataset, batch_size=32, num_workers=4,
                        sampler=RandomSampler(dataset), persistent_workers=True,
                        worker_init_fn=partial(obs_utils_init_worker, obs_modality=obs_modality))
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i == 0:
            print("batch keys:", sorted(batch.keys()), flush=True)
        if i == 9:
            break
    print(f"first 10 batches with 4 workers: {(time.time()-t0)/10*1000:.0f} ms/batch (incl. worker startup)", flush=True)
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i == 19:
            break
    print(f"next 20 batches (warm workers): {(time.time()-t0)/20*1000:.0f} ms/batch", flush=True)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

# Task 1 — DataLoader Implementation

## Single-GPU DataLoader

Please refer src/dataloaders.py for the single-GPU dataloader implementation.

## Multi-GPU DataLoader 

To parallelize training across multiple GPUs, one can adopt PyTorch’s
**Distributed Data Parallel (DDP)** execution model and design the data
loading pipeline to scale correctly and efficiently across processes. 

---

### Distributed Training Model (DDP)

We follow the standard **one-process-per-GPU** paradigm:

- Each GPU is driven by a single training process
- Each process owns:
  - a duplicated model
  - its own optimizer
  - its own `DataLoader`
- Gradients are synchronized across processes during backpropagation

One process per GPU avoids GPU memory contention. 

---

### Dataset Sharding with `DistributedSampler`

To ensure that each GPU processes a unique subset of data, we use
`DistributedSampler`:

- The dataset is partitioned across all training processes (GPUs)
- Each process is identified by a unique `rank`
- No data duplication occurs across GPUs
- `shuffle=False` is set on the `DataLoader`, since shuffling is handled by the sampler

For correct stochastic behavior, the sampler is reshuffled at the start of
each epoch:

```python
sampler.set_epoch(epoch)
```

This step is critical for proper shuffling in DDP.

---

### Batch Size 

The batch_size parameter specifies the per-GPU batch size.

The effective global batch size is: global_batch_size = batch_size × world_size

This separation allows straightforward scaling across different GPU counts.

---

### DataLoader Worker Processes (`num_workers`)

Each training process spawns multiple **CPU-only DataLoader worker processes**:

- `num_workers` specifies the number of data-loading worker processes **per GPU** (per DDP process)
- These workers are responsible for:
  - disk I/O
  - resizing and geometric augmentation
- DataLoader workers do **not** perform GPU computation and do **not** participate in gradient synchronization

This design allows CPU-side data preparation to overlap with GPU computation, reducing GPU idle time and improving overall training throughput.

---

### Prefetching and Throughput Optimization

To reduce GPU idle time, batch prefetching is enabled:

- `prefetch_factor` controls how many batches each DataLoader worker prepares in advance
- Prefetching allows CPU data loading and preprocessing to overlap with GPU computation
- This is especially important in DDP, where slow data loading on one GPU can stall all
  others at synchronization points

Additional performance-related options include:

- `pin_memory=True` for faster host-to-device transfers
- `drop_last=True` to ensure consistent batch shapes across ranks


Minimal pseudocode (use this in your trainer)

```python
dist.init_process_group(backend='nccl') # signal we are doing distributed training
torch.cuda.set_device(local_rank) # bind process to one specific GPU
loader, sampler = make_distributed_dataloader(
    dataset,
    batch_size=per_gpu_batch,
    num_replicas=world_size,
    rank=rank,
    num_workers=workers_per_proc,
)
for epoch in range(epochs):
    sampler.set_epoch(epoch) # sampler shuffles the dataset each epoch
    for batch in loader: # each GPU see different data
        imgs = batch['pixel_values'].to(device, non_blocking=True)
        # forward/backward/step
```
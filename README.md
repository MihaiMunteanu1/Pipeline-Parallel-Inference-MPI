# Pipeline-Parallel Inference with MPI

Neural network inference split across the nodes of a virtual Rocks 6.2 cluster,
on CPU only, in C with Open MPI. Each node runs some of the model's layers and
sends the intermediate tensors to the next node.

The main model is `bge-base-en-v1.5`, used for semantic search over 1,000 news texts.

## Results

Processing 50 texts with `bge-base-en-v1.5`, mean of 3 runs:

| Compute nodes | Time | Speed-up | Efficiency |
|---|---|---|---|
| 1 | 286.1 s | - | - |
| 2 | 206.2 s | 1.39x | 69% |
| 4 | 132.6 s | 2.16x | 54% |

Parallelism speeds up processing many texts, not a single question. One question
took about 1.34 s on 4 nodes and 0.65 s on 1 node, in a single measurement.
All VMs run on one laptop, so these numbers are indicative.

Details are in the [paper](Pipeline_Parallelism_Mihai_Munteanu.pdf).

## Files

- `embed_mpi.c`: BERT encoder, search and live view
- `run_mpi.c`: Llama 2-style generation, based on llama2.c
- `export_bge.py`, `export_minilm.py`: export the models in Google Colab
- `bench_scale.sh`, `bench_embed.sh`: speed measurements
- `ask.sh`, `live.sh`: demo

## Run

Needs Open MPI, plus a `hosts4` file listing the 4 compute nodes.
`bge_data.tar` is in the `v2` release.

    tar xf bge_data.tar
    mpicc -std=gnu99 -O3 -ffast-math -o embed_mpi embed_mpi.c -lm
    N="--mca btl tcp,self --mca btl_tcp_if_include eth0"
    mpirun -np 4 --hostfile hosts4 $N ./embed_mpi bge.bin vocab.txt -f texts.txt -o vectors.bin -w 8
    sh ask.sh "oil prices are rising"
    sh live.sh

`run_mpi.c` is based on [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy (MIT License).

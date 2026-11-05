# Asks a question on 4 compute nodes and shows live what every node does.
# Usage: sh ask.sh "oil prices are rising"
N="--mca btl tcp,self --mca btl_tcp_if_include eth0"
P="Represent this sentence for searching relevant passages:"
mpirun -np 4 --hostfile hosts4 $N ./embed_mpi bge.bin vocab.txt -f texts.txt -v vectors.bin -p "$P" -t 1 -q "$1"

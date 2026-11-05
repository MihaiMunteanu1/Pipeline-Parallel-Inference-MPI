# Processes 8 texts on 4 compute nodes and shows live what every node does.
N="--mca btl tcp,self --mca btl_tcp_if_include eth0"
head -8 texts.txt > t8.txt
mpirun -np 4 --hostfile hosts4 $N ./embed_mpi bge.bin vocab.txt -f t8.txt -w 8 -t 1

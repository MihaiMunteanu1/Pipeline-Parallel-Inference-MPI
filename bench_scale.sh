# Compares 1, 2 and 4 compute nodes on the same texts, 3 times each.
N="--mca btl tcp,self --mca btl_tcp_if_include eth0"
M="bge.bin vocab.txt -f t50.txt"
head -50 texts.txt > t50.txt
for i in 1 2 3; do
  mpirun -np 1 --host compute-0-0 $N ./embed_mpi $M 2>&1 | grep ranks:
  mpirun -np 2 --hostfile hosts $N ./embed_mpi $M -w 4 2>&1 | grep ranks:
  mpirun -np 4 --hostfile hosts4 $N ./embed_mpi $M -w 8 2>&1 | grep ranks:
done

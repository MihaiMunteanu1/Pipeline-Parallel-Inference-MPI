N="--mca btl tcp,self --mca btl_tcp_if_include eth0"
head -200 texts.txt > t200.txt
for i in 1 2 3; do
  mpirun -np 1 --host compute-0-0 $N ./embed_mpi minilm.bin vocab.txt -f t200.txt 2>&1 | grep ranks:
  mpirun -np 2 --hostfile hosts $N ./embed_mpi minilm.bin vocab.txt -f t200.txt -w 1 2>&1 | grep ranks:
  mpirun -np 2 --hostfile hosts $N ./embed_mpi minilm.bin vocab.txt -f t200.txt -w 4 2>&1 | grep ranks:
done

N="--mca btl tcp,self --mca btl_tcp_if_include eth0"
M="stories15M.bin -s 42 -n 200"
for i in 1 2 3; do
  mpirun -np 1 --host compute-0-0 $N ./run_mpi $M 2>&1 >/dev/null | grep ranks:
  mpirun -np 1 --host compute-0-0 $N ./run_mpi $M -b 2 2>&1 >/dev/null | grep ranks:
  mpirun -np 2 --hostfile hosts $N ./run_mpi $M 2>&1 >/dev/null | grep ranks:
  mpirun -np 2 --hostfile hosts $N ./run_mpi $M -b 2 2>&1 >/dev/null | grep ranks:
  mpirun -np 2 --hostfile hosts $N ./run_mpi $M -b 2 -l 0,6 2>&1 >/dev/null | grep ranks:
done

#!/bin/bash
#SBATCH -A IscrC_RLSPACE
#SBATCH -p m100_usr_prod
#SBATCH --time 00:30:00         # format: HH:MM:SS
#SBATCH --gres=gpu:4            # 1 gpus per node out of 4
#SBATCH --mem=64000             # memory per node out of 246000MB
#SBATCH --job-name=my_batch_job1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alessandro.zavoli@uniroma1.it
#SBATCH --nodes=1             # nodes -N 1
#SBATCH --ntasks-per-node=1   # MPI tasks/node
#SBATCH --cpus-per-task=128   # OpenMP threads/task

#mpirun ./myexecutable       #in case you compiled with spectrum-mpi
#OR
#srun ./myexecutable         #in all the other cases

#SBATCH --output="slurm-%A.out"


 

module load profile/deeplrn
#module load autoload cineca-ai/2.1.0
module load autoload ray/2.0.0--gnu--10.3.0 
source /m100/home/userexternal/azavoli0/RAYWorks2/rayvenv/bin/activate

# echo "srun  -n2 su 2 nodo"
# srun  -n 2 python a2c-cart.py


rllib train --run=PPO --env=CartPole-v0 --config '{"num_workers": 4,"num_gpus":1}'
#srun  -n 1 rllib train --ray-num-cpus $SLURM_CPUS_PER_TASK --ray-num-gpus 1 --run=PPO --env=CartPole-v0 --config '{"num_workers":4,"num_gpus":1}'
#srun  -n 1 /cineca/prod/opt/libraries/ray/1.0.0/python--3.8.2/bin/python3 -u `which rllib` train --ray-num-cpus 256 --ray-num-gpus 4 --run=IMPALA --env=CartPo$



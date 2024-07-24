#!/bin/bash
part=klab-cpu
node=1
cpu=16
ram=15
gpu=0
read -p "Enter the partition name (gpu, workq, klab-cpu, klab-l40s, default $part): " PARTITION 
read -p "Enter the number of nodes (1, N, default $node): " NODES
read -p "Enter the number of CPUS (1, C, default $cpu): " CPUS
read -p "Enter the required RAM (in GB, default $ram): " RAM
read -p "Enter the number of GPUs (1, M, default $gpu): " GPU
PARTITION=${PARTITION:-$part}
NODES=${NODES:-$node}
CPUS=${CPUS:-$cpu}
RAM=${RAM:-$ram}
GPU=${GPU:-$gpu}
echo "Submitting a job on partition" $PARTITION "with" $NODES "nodes", $CPUS "cpus" and $RAM "GB of RAM" 
echo "Interactive sessions are limited to 4 hours" 
if [[ $PARTITION == "gpu" ]]
then
            salloc -p $PARTITION -n $NODES -c $CPUS --mem "$RAM"G --gres=gpu:A100:$GPU -t 04:00:00  srun  --pty bash	            
fi

if [[ $PARTITION == "klab-l40s" ]]
then
	    salloc -p $PARTITION -n $NODES -c $CPUS --mem "$RAM"G --gres=gpu:$GPU -t 04:00:00 srun --pty bash
fi

if [[ $PARTITION == "workq" || $PARTITION == "klab-cpu" ]]
then
            salloc -p $PARTITION -n $NODES -c $CPUS --mem "$RAM"G -t 04:00:00  srun  --pty bash	            
fi

if [[ $PARTITION != "workq"  &&  $PARTITION != "gpu" && $PARTITION != "klab-cpu" && $PARTITION != "klab-l40s" ]]
then

       echo "The parition has to be either workq, gpu, klab-cpu or klab-l40s"
fi        

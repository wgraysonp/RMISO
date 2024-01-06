#!/bin/bash

EPOCHS=200
N_ITER=50

DATASET="a9a"

SAMPLER="random_walk"
MODEL="one_layer"

GRAPH_EDGES=99
GRAPH_NODES=50
GRAPH_TOPO="complete"
RADIUS=0.3

RMISO_LRS=(10 5 1)
RMISO_DELTS=(1)
RMISO_RHOS=(250 50 10)

ADAGRAD_LRS=(0.05 0.01 0.005)

SGD_DECAY_LRS=(0.5 0.1 0.05 0.01)

SGD_LRS=(0.1 0.05 0.01 0.005)

SAG_LRS=(10 5 1)
SAG_TAUS=(50 2500)

echo "running $MODEL with $GRAPH_NODES nodes $GRAPH_EDGES edges $GRAPH_TOPO topology"

for lr in ${RMISO_LRS[@]}; do
	for rho in ${RMISO_RHOS[@]}; do
		echo "optim: rmiso"
		echo "learning rate: $lr"
		echo "rho: $rho"
		python3 main.py  --epochs $EPOCHS --epoch_length $N_ITER --sep_classes --sampling_algorithm $SAMPLER --radius $RADIUS --graph_size $GRAPH_NODES --graph_edges $GRAPH_EDGES --graph_topo $GRAPH_TOPO --model $MODEL --optim rmiso --init_optimizer --lr $lr --rho $rho --dataset $DATASET --save
	done
done

for lr in ${RMISO_LRS[@]}; do
        for rho in ${RMISO_RHOS[@]}; do
                echo "optim: rmiso"
                echo "learning rate: $lr"
                echo "rho: $rho"
                python3 main.py --dynamic_step  --epochs $EPOCHS --epoch_length $N_ITER --sep_classes --sampling_algorithm $SAMPLER --radius $RADIUS --graph_size $GRAPH_NODES --graph_edges $GRAPH_EDGES --graph_topo $GRAPH_TOPO --model $MODEL --optim rmiso --init_optimizer --lr $lr --rho $rho --dataset $DATASET --save
        done
done

for lr in ${SAG_LRS[@]}; do
	for tau in ${SAG_TAUS[@]}; do
        	echo "optim: sag"
      	 	echo "learning rate: $lr"
        	python3 main.py --epochs $EPOCHS --epoch_length $N_ITER --sep_classes --sampling_algorithm $SAMPLER --radius $RADIUS --graph_size $GRAPH_NODES --graph_edges $GRAPH_EDGES --graph_topo $GRAPH_TOPO --model $MODEL  --optim mcsag --init_optimizer --lr $lr --tau $tau --dynamic_step --dataset $DATASET  --save
	done
done

for lr in ${SGD_LRS[@]}; do
        echo "optim: sgd"
        echo "learning rate: $lr"
        python3 main.py --epochs $EPOCHS --epoch_length $N_ITER --sep_classes --sampling_algorithm $SAMPLER --radius $RADIUS --graph_size $GRAPH_NODES --graph_edges $GRAPH_EDGES --graph_topo $GRAPH_TOPO --model $MODEL --optim sgd --lr $lr --dataset $DATASET --save
done

for lr in ${SGD_DECAY_LRS[@]}; do
        echo "optim: sgd"
        echo "learning rate: $lr"
        python3 main.py --epochs $EPOCHS --epoch_length $N_ITER --sep_classes --sampling_algorithm $SAMPLER --radius $RADIUS --graph_size $GRAPH_NODES --graph_edges $GRAPH_EDGES --graph_topo $GRAPH_TOPO --model $MODEL --optim sgd --lr $lr --lr_decay --dataset $DATASET --save
done

for lr in ${ADAGRAD_LRS[@]}; do
        echo "optim: sgd"
        echo "learning rate: $lr"
        python3 main.py --epochs $EPOCHS --epoch_length $N_ITER --sep_classes --sampling_algorithm $SAMPLER --radius $RADIUS --graph_size $GRAPH_NODES --graph_edges $GRAPH_EDGES --graph_topo $GRAPH_TOPO --model $MODEL --optim adagrad --lr $lr --dataset $DATASET --save
done

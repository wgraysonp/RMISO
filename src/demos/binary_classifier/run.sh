#!/bin/bash

EPOCHS=100
N_ITER=10

SAMPLER="metropolis_hastings"
MODEL="one_layer"

GRAPH_EDGES=99
GRAPH_NODES=50
GRAPH_TOPO="geometric"
RADIUS=0.4

RMISO_LRS=(20 10 5 1)
RMISO_DELTS=(1 1e-5)
RMISO_RHOS=(1)

ADAM_LRS=(0.01)
ADAM_BETA1=0.9
ADAM_BETA2=0.99

SGD_LRS=(10 1 0.1 0.01)

SAG_LRS=(3)
SAG_TAU=50
SAG_DELTS=(1)

echo "running $MODEL with $GRAPH_NODES nodes $GRAPH_EDGES edges $GRAPH_TOPO topology"

#for lr in ${RMISO_LRS[@]}; do
#	for delta in ${RMISO_DELTS[@]}; do
#		echo "optim: rmiso"
#		echo "learning rate: $lr"
#		echo "delta: $delta"
#		python3 main.py --epochs $EPOCHS --epoch_length $N_ITER --sep_classes --sampling_algorithm $SAMPLER --radius $RADIUS --graph_size $GRAPH_NODES --graph_edges $GRAPH_EDGES --graph_topo $GRAPH_TOPO --model $MODEL --optim rmiso --init_optimizer --lr $lr --delta $delta --dynamic_step --save
#	done
#done

#for lr in ${SAG_LRS[@]}; do
	#for delta in ${SAG_DELTS[@]}; do
        	#echo "optim: sag"
      	 	#echo "learning rate: $lr"
        	#python3 main.py --epochs $EPOCHS --epoch_length $N_ITER --sep_classes --sampling_algorithm $SAMPLER --radius $RADIUS --graph_size $GRAPH_NODES --graph_edges $GRAPH_EDGES --graph_topo $GRAPH_TOPO --model $MODEL --delta $delta  --optim mcsag --init_optimizer --lr $lr --tau $SAG_TAU --dynamic_step  --save
	#done
#done

#for lr in ${SGD_LRS[@]}; do
#        echo "optim: sgd"
#        echo "learning rate: $lr"
#        python3 main.py --epochs $EPOCHS --epoch_length $N_ITER --sep_classes --sampling_algorithm $SAMPLER --radius $RADIUS --graph_size $GRAPH_NODES --graph_edges $GRAPH_EDGES --graph_topo $GRAPH_TOPO --model $MODEL --optim sgd --lr $lr --save
#done

for lr in ${ADAM_LRS[@]}; do
        echo "optim: adam"
        echo "learning rate: $lr"
        python3 main.py --epochs $EPOCHS --epoch_length $N_ITER --sep_classes --sampling_algorithm $SAMPLER --radius $RADIUS --graph_size $GRAPH_NODES --graph_edges $GRAPH_EDGES --graph_topo $GRAPH_TOPO --model $MODEL --optim adam --lr $lr --beta1 $ADAM_BETA1 --beta2 $ADAM_BETA2 --save
done

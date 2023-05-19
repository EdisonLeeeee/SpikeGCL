## SpikeGCL (Under review)
This is a PyTorch implementation of SpikeGCL from the paper "A Graph is Worth 1-bit Spikes: When Graph Contrastive Learning Meets Spiking Neural Networks".

## Environments
+ numpy == 1.23.3
+ torch == 1.8+cu111
+ torch-cluster == 1.6.1
+ torch_geometric == 2.3.0
+ torch-scatter == 2.1.1
+ torch-sparse == 0.6.17
+ CUDA 11.1
+ cuDNN 8.0.5


## Reproduction

+ Cora
```
python main.py --dataset Cora --threshold 5e-4 --outs 4 --bn --timesteps 64
```
+ Citeseer
```
python main.py --dataset Citeseer --threshold 5e-3 --bn
```
+ Pubmed
```
python main.py --dataset Pubmed --threshold 5e-2 --bn
```
+ Computers
```
python main.py --dataset Computers --threshold 5e-2 --outs 32 --bn 
```
+ Photo 
```
python main.py --dataset Photo --threshold 0.01 --timesteps 50 --bn --outs 32 --lr 0.001
```
+ CS
```
python main.py --dataset CS --threshold 5e-2 --outs 64 --timesteps 25 --dropout 0.1 --dropedge 0.2 --lr 0.001 --bn
```
+ Physics 
```
python main.py --dataset Physics --timesteps 50 --outs 64 --lr 0.1
```
+ Ogbn-arXiv
```
python main.py --dataset ogbn-arxiv --timesteps 25 --outs 64 --lr 0.1 --threshold 0.01 --bn
```
+ Ogbn-MAG
```
python main.py --dataset ogbn-mag --timesteps 10 --outs 32 --lr 0.001 --threshold 0.01 --bn
```
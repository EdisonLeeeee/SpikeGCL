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
python main.py --dataset Cora --threshold 5e-4 --outs 4 --bn --T 64
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
python main.py --dataset Photo --threshold 5e-2 --T 16 --bn --outs 8
```
+ CS
```
python main.py --dataset CS --threshold 5e-1 --outs 32 --T 64 --dropout 0. --bn
```
+ Physics 
```
python main.py --dataset Physics --T 50 --outs 64 --lr 0.1
```
+ Ogbn-arXiv
```
python main.py --dataset ogbn-arxiv --T 25 --outs 64 --lr 0.1 --threshold 0.01 --bn
```
+ Ogbn-MAG
```
python main.py --dataset ogbn-mag --T 10 --outs 32 --lr 0.001 --threshold 0.01 --bn
```
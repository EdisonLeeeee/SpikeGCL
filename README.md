## SpikeGCL (Under review)
This is a PyTorch implementation of SpikeGCL from the paper "A Graph is Worth 1-bit Spikes: When Graph Contrastive Learning Meets Spiking Neural Networks".

<div align="center">
  <img src="imgs/spikegcl.png"/>
</div>

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
python main.py --dataset Citeseer --threshold 5e-3 --bn --T 64
```
+ Pubmed
```
python main.py --dataset Pubmed --threshold 5e-2 --bn --T 32
```
+ Computers
```
python main.py --dataset Computers --threshold 5e-2 --outs 32 --bn --T 25
```
+ Photo 
```
python main.py --dataset Photo --threshold 5e-2 --T 15 --bn --outs 8
```
+ CS
```
python main.py --dataset CS --threshold 5e-1 --outs 32 --T 60 --dropout 0. --bn
```
+ Physics 
```
python main.py --dataset Physics --T 25 --outs 16 --bn --margin 1.0 --threshold 5e-2
```
+ Ogbn-arXiv
```
python main.py --dataset ogbn-arxiv --T 15 --outs 32 --threshold 5e-2 --bn
```
+ Ogbn-MAG
```
python main.py --dataset ogbn-mag --T 15 --outs 32 --threshold 5e-3 --bn
```
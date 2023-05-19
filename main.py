import argparse

import numpy as np
import torch
from spikegcl.evaluate import test
from spikegcl.dataset import get_dataset
from spikegcl.model import SpikeGCL
from spikegcl.utils import tab_printer
from torch_geometric import seed_everything
from torch_geometric.logging import log


def read_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="~/public_data/pyg_data", help="Data folder"
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="Pubmed",
        help="Datasets (Photo, Computers, CS, Physics, Cora, Citeseer, Pubmed, ogbn-arxiv, ogbn-mag). (default: Pubmed)",
    )
    parser.add_argument(
        "--hids",
        type=int,
        default=64,
        help="Hidden units for each layer. (default: 64)",
    )
    parser.add_argument(
        "--outs",
        type=int,
        default=64,
        help="Out_channels for final embedding. (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for training. (default: 1e-3)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs. (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help="Random seed for model. (default: 2023)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Smooth factor for surrogate learning. (default: 2.0)",
    )
    parser.add_argument(
        "--surrogate",
        nargs="?",
        default="sigmoid",
        help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')",
    )
    parser.add_argument(
        "--neuron",
        nargs="?",
        default="PLIF",
        help="Spiking neuron used for training. (IF, LIF, PLIF). (default: PLIF)",
    )
    parser.add_argument(
        "--reset",
        nargs="?",
        default="subtract",
        help="Ways to reset spiking neuron. (zero, subtract). (default: subtract)",
    )
    parser.add_argument(
        "--act",
        nargs="?",
        default="elu",
        help="Activation function. (relu, elu, None). (default: elu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5e-3,
        help="Voltage threshold in spiking neuron. (default: 5e-3)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=32,
        help="Spiking Time steps. (default: 32)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability. (default: 0.5)"
    )
    parser.add_argument(
        "--dropedge",
        type=float,
        default=0.2,
        help="Edge dropout probability. (default: 0.2)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Margin used in ranking loss. (default: 0.0)",
    )
    parser.add_argument('--bn', action='store_true',
                    help='Whether to use batch normalization. (default: False)')
    try:
        args = parser.parse_args()
        tab_printer(args)
        return args
    except:
        parser.print_help()
        exit(0)


args = read_parser()
seed_everything(args.seed)

data = get_dataset(
    root=args.root,
    dataset=args.dataset,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = SpikeGCL(
    data.x.size(1),
    args.hids,
    args.outs,
    args.timesteps,
    args.alpha,
    args.surrogate,
    args.threshold,
    args.neuron,
    args.reset,
    args.act,
    args.dropedge,
    args.dropout,
    bn=args.bn,
)
print(model)
model, data = model.to(device), data.to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay=0.0, lr=args.lr)

def train():
    model.train()
    optimizer.zero_grad()
    loss_total = 0.0
    z1s, z2s = model(data.x, data.edge_index, data.edge_attr)
    for z1, z2 in zip(z1s, z2s):
        loss = model.loss(z1, z2, args.margin)
        loss.backward()
        loss_total += loss.item()
    optimizer.step()
    return loss_total


best_val_acc = final_test_acc = 0

for epoch in range(1, args.epochs + 1):
    loss = train()
    model.eval()
    with torch.no_grad():
        embeds = model.encode(data.x, data.edge_index, data.edge_attr)
        embeds = torch.cat(embeds, dim=-1)
        print("=" * 100)
        print(f"Firing rate: {embeds.mean().item():.2%}")
        print("=" * 100)
    val_accs, test_accs = test(embeds, data, data.num_classes)
    val_acc = np.mean(val_accs)
    test_acc = np.mean(test_accs)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        final_test_acc = test_acc

    log(Epoch=epoch, Loss=loss, val_acc=val_acc, test_acc=test_acc, best=final_test_acc)

log(Final_Acc=final_test_acc)

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from sklearn.metrics import roc_auc_score

def get_dataset(root, dataset, num_val=0.1, num_test=0.8):
    if dataset in {"Photo", "Computers"}:
        dataset = Amazon(root, dataset, transform=T.ToUndirected())
        data = dataset[0]
        data = T.RandomNodeSplit(num_val=num_val, num_test=num_test)(data)
    elif dataset in {"CS", "Physics"}:
        dataset = Coauthor(root, dataset, transform=T.ToUndirected())
        data = dataset[0]
        data = T.RandomNodeSplit(num_val=num_val, num_test=num_test)(data)
    elif dataset in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(name=dataset, root=root, split="full")
        data = dataset[0]
    elif dataset in ["ogbn-arxiv", "ogbn-mag"]:
        from ogb.nodeproppred import PygNodePropPredDataset
        if dataset == "ogbn-mag":
            dataset = PygNodePropPredDataset(
            root=root, name=dataset, transform=T.ToUndirected()
        )
            # We are only interested in paper <-> paper relations.
            rel_data = dataset[0]
            
            data = Data(
                x=rel_data.x_dict["paper"],
                edge_index=rel_data.edge_index_dict[("paper", "cites", "paper")],
                y=rel_data.y_dict["paper"],
            )
            data.y = data.y.squeeze()
            train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
            val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
            test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
            split_idx = dataset.get_idx_split()
            train_mask[split_idx['train']['paper']] = True
            val_mask[split_idx["valid"]['paper']] = True
            test_mask[split_idx["test"]['paper']] = True
        else:
            dataset = PygNodePropPredDataset(
            root=root, name=dataset, transform=T.ToUndirected()
        )
            data = dataset[0]
            data.y = data.y.squeeze()
            train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
            val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
            test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
            split_idx = dataset.get_idx_split()
            train_mask[split_idx["train"]] = True
            val_mask[split_idx["valid"]] = True
            test_mask[split_idx["test"]] = True
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    else:
        raise ValueError(dataset)
    data.num_classes = dataset.num_classes
    return data

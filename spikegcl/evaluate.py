import torch
import torch.nn as nn
import torch.nn.functional as F


def test(embeds, data, num_classes):
    if not isinstance(embeds, torch.Tensor):
        embeds = torch.cat(embeds, dim=1)
        
    return node_cls_downstream_task_eval(
        input_emb=embeds, data=data, num_classes=num_classes,
        lr=0.01, wd=5e-4,
        cls_epochs=100, cls_runs=5, device=embeds.device)


def eval_acc(model, x, y):
    model.eval()
    with torch.no_grad():
        output = model(x)
        y_pred = torch.argmax(output, dim=1).squeeze(-1)

        return (y_pred == y).float().mean().item()


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        return x.log_softmax(dim=-1)

    def reset_parameters(self):
        self.linear.reset_parameters()


def train_cls(cls, x, y, train_mask, val_mask, test_mask,
              lr=1e-2, weight_decay=1e-5, epochs=100):
    cls.reset_parameters()
    optimizer = torch.optim.AdamW(
        cls.parameters(), lr=lr, weight_decay=weight_decay)

    train_x, train_y = x[train_mask], y[train_mask]
    val_x, val_y = x[val_mask], y[val_mask]
    test_x, test_y = x[test_mask], y[test_mask]

    best_val_acc, best_test_acc = 0.0, 0.0
    for _ in range(epochs):
        cls.train()
        optimizer.zero_grad()

        output = cls(train_x)
        loss = F.nll_loss(output, train_y)
        loss.backward()
        optimizer.step()

        val_acc, test_acc = eval_acc(cls, val_x, val_y), eval_acc(cls, test_x, test_y)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    return best_val_acc, best_test_acc


def node_cls_downstream_task_eval(input_emb, data, num_classes,
                                  lr, wd, cls_epochs=100, 
                                  cls_runs=10, device="cpu"):
    all_val_acc, all_test_acc = [], []

    # input_emb = F.normalize(input_emb, dim=1)    # l2 normalize
    gnn_emb_dim = input_emb.size(1)

    classifier = Classifier(gnn_emb_dim, num_classes).to(device)

    for _ in range(cls_runs):

        best_val_acc, best_test_acc = train_cls(
            classifier, input_emb, data.y,
            data.train_mask, data.val_mask, data.test_mask,
            lr=lr, weight_decay=wd, epochs=cls_epochs)

        all_val_acc.append(best_val_acc)
        all_test_acc.append(best_test_acc)

    return all_val_acc, all_test_acc

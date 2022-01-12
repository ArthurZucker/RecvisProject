import torch 
import torch.nn as nn

class CrossCorrelationMatrixLoss(nn.Module):

    def __init__(self,lmbda) -> None:
        super().__init__()
        self.lmbda = lmbda

    def forward(self,z1,z2):
        # Normalize the projector's output across the batch
        norm_z1 = (z1 - torch.mean(z1,0)) / torch.std(z1,0)
        norm_z2 = (z2 - torch.mean(z1,0)) / torch.std(z1,0)

        # Cross correlation matrix
        batch_size = z1.size(0)
        # average over the batch size to get a 2D correlation matrix 
        cc_M = torch.einsum("bi,bj->ij", (norm_z1, norm_z2)) / batch_size
        self.cc_M = cc_M.detach().cpu().numpy()
        # Invariance loss
        diag = torch.diagonal(cc_M)
        invariance_loss = torch.sum(((torch.ones_like(diag) - diag) ** 2))

        # Zero out the diag elements and flatten the matrix to compute the loss
        cc_M.fill_diagonal_(0)
        redundancy_loss = torch.sum((cc_M.flatten() ** 2)) # TODO try L1 loss to push to 0 rather than closse to 0  
        loss = invariance_loss + self.lmbda * redundancy_loss

        return loss

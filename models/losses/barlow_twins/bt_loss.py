import torch 

def BarlowTwinsLoss(z1, z2, lmbda):

    # Normalize the projector's output across the batch
    norm_z1 = (z1 - z1.mean(0)) / z1.std(0)
    norm_z2 = (z2 - z2.mean(0)) / z2.std(0)

    # Cross correlation matrix
    batch_size = z1.size(0)
    cc_M = torch.einsum("bi,bj->ij", (norm_z1, norm_z2)) / batch_size

    # Invariance loss
    diag = torch.diagonal(cc_M)
    invariance_loss = ((torch.ones_like(diag) - diag) ** 2).sum()

    # Zero out the diag elements and flatten the matrix to compute the loss
    cc_M.fill_diagonal_(0)
    redundancy_loss = (cc_M.flatten() ** 2).sum()
    loss = invariance_loss + lmbda * redundancy_loss

    return loss
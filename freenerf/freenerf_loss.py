import torch

def occ_reg_loss_fn(rgb, density, reg_range=10, wb_prior=False, wb_range=20):
    """
    Computes the occlusion regularization loss.

    Args:
        rgb (torch.Tensor): The RGB rays/images.
        density (torch.Tensor): The current density map estimate.
        reg_range (int): The number of initial intervals to include in the regularization mask.
        wb_prior (bool): If True, a prior based on the assumption of white or black backgrounds is used.
        wb_range (int): The range of RGB values considered to be a white or black background.

    Returns:
        float: The mean occlusion loss within the specified regularization range and white/black background region.
    """
    # Compute the mean RGB value over the last dimension
    rgb_mean = rgb.mean(dim=-1)

    # Compute a mask for the white/black background region if using a prior
    if wb_prior:
        white_mask = torch.where(rgb_mean > 0.99, torch.tensor(1), torch.tensor(0))  # A naive way to locate white background
        black_mask = torch.where(rgb_mean < 0.01, torch.tensor(1), torch.tensor(0))  # A naive way to locate black background
        rgb_mask = (white_mask + black_mask).to(torch.float32)  # White or black background
        rgb_mask[:, wb_range:] = 0  # White or black background range
    else:
        rgb_mask = torch.zeros_like(rgb_mean) # la maschera ha tutti gli elementi uguali a 0 senza prior

    # Create a mask for the general regularization region
    if reg_range > 0:
        rgb_mask[:, :reg_range] = 1  # Penalize the points in reg_range close to the camera

    # Compute the density-weighted loss within the regularization and white/black background mask
    return torch.mean(density * rgb_mask)
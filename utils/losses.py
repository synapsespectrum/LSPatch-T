import torch
import torch.nn as nn


def masked_frequency_loss(pred, target, mask=None) -> torch.Tensor:
    """
    Compute the masked frequency domain loss.

    Args:
        pred (torch.Tensor): Predicted values [bs x num_patch x n_vars x patch_len]
        target (torch.Tensor): Ground truth values [bs x num_patch x n_vars x patch_len]
        mask (torch.Tensor): Mask for valid data points [bs x num_patch x n_vars]

    Returns:
        torch.Tensor: Masked frequency domain loss
    """
    # Convert to frequency domain
    freq_gt = mag_scaling(torch.fft.fftn(target.float(), dim=(-3, -2, -1), norm='ortho'))
    freq_pred = mag_scaling(torch.fft.fftn(pred.float(), dim=(-3, -2, -1), norm='ortho'))

    # Compute log difference
    freq_dis = torch.log1p(torch.abs(freq_gt - freq_pred))

    # Remove NaN values
    freq_dis[torch.isnan(freq_dis)] = 0.0

    # Average over patch length (last dimension)
    freq_loss = freq_dis.mean(dim=-1)  # [bs x num_patch x n_vars]

    if mask is None:
        mask = torch.ones_like(freq_loss)
    # Apply mask and calculate mean
    freq_loss = (freq_loss * mask).sum() / mask.sum()

    return freq_loss


def mag_scaling(x):
    """
    Apply magnitude scaling to the complex tensor.

    Args:
        x (torch.Tensor): Complex tensor in frequency domain

    Returns:
        torch.Tensor: Scaled complex tensor
    """
    magnitude = torch.abs(x)
    phase = torch.angle(x)
    magnitude_scaled = torch.log1p(magnitude)
    return magnitude_scaled * torch.exp(1j * phase)


class MaskedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, mask) -> torch.Tensor:
        """
        pred:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len]
        mask:    [bs x num_patch x n_vars x patch_len] or broadcastable
        """
        # Calculate MSE
        mse_loss = (pred - target) ** 2
        mse_loss = mse_loss.mean(dim=-1)  # Average over patch length
        mse_loss = (mse_loss * mask).sum() / mask.sum()  # Apply mask and calculate mean

        # Calculate frequency loss
        freq_loss = masked_frequency_loss(pred, target, mask)

        # Combine losses
        loss = self.alpha * mse_loss + (1 - self.alpha) * freq_loss

        return loss

    def plot_frequency_domain(self, pred, gt, sample_index=0):
        """
        Plot the magnitude of the frequency domain representation for prediction and ground truth.

        Args:
            pred (torch.Tensor): Predicted values [bs x num_patch x n_vars x patch_len]
            gt (torch.Tensor): Ground truth values [bs x num_patch x n_vars x patch_len]
            sample_index (int): Index of the sample to plot (default is 0)

        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        import matplotlib.pyplot as plt

        # Convert to frequency domain
        freq_gt = self.mag_scaling(torch.fft.fftn(gt.float(), dim=(-3, -2, -1), norm='ortho'))
        freq_pred = self.mag_scaling(torch.fft.fftn(pred.float(), dim=(-3, -2, -1), norm='ortho'))

        # Get magnitude
        mag_gt = torch.abs(freq_gt[sample_index]).cpu().numpy()
        mag_pred = torch.abs(freq_pred[sample_index]).cpu().numpy()

        # Create figure
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Frequency Domain Magnitude Comparison')

        # Plot for each variable
        for i in range(min(3, mag_gt.shape[0])):  # Plot up to 3 variables
            axs[0, i].imshow(mag_gt[i], cmap='viridis', aspect='auto')
            axs[0, i].set_title(f'Ground Truth - Variable {i + 1}')
            axs[0, i].set_xlabel('Patch Length')
            axs[0, i].set_ylabel('Num Patch')

            axs[1, i].imshow(mag_pred[i], cmap='viridis', aspect='auto')
            axs[1, i].set_title(f'Prediction - Variable {i + 1}')
            axs[1, i].set_xlabel('Patch Length')
            axs[1, i].set_ylabel('Num Patch')

        plt.tight_layout()
        return fig


class DownstreamLoss(nn.Module):
    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target) -> torch.Tensor:
        """
        preds:    # [Batch, pred_len, n_vars]
        targets:  # [Batch, pred_len, n_vars]
        """
        # Calculate MSE
        mse_loss = self.mse_loss(pred, target)

        # Calculate frequency loss
        freq_loss = masked_frequency_loss(pred, target)

        # Combine losses
        loss = mse_loss + freq_loss

        return loss

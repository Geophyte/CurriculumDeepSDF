import torch.nn as nn
import torch
import copy
from tqdm import tqdm
from utils import utils_deepsdf

"""
Model based on the paper 'DeepSDF'.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SDFModel(torch.nn.Module):
    def __init__(self, num_layers, skip_connections, latent_size, inner_dim=512, output_dim=1):
        """
        SDF model for multiple shapes.
        Args:
            input_dim: 128 for latent space + 3 points = 131
        """
        super(SDFModel, self).__init__()
        self.alpha = 1.0

        # Num layers of the entire network
        self.num_layers = num_layers

        # If skip connections, add the input to one of the inner layers
        self.skip_connections = skip_connections

        self.latent_size = latent_size

        # Dimension of the input space (3D coordinates)
        dim_coords = 3
        self.input_dim = self.latent_size + dim_coords
        self.inner_dim = inner_dim

        # Copy input size to calculate the skip tensor size
        self.skip_tensor_dim = copy.copy(self.input_dim)

        # Compute how many layers are not Sequential
        num_extra_layers = 2 if (self.skip_connections) else 1

        # Add sequential layers
        layers = []
        for _ in range(num_layers - num_extra_layers):
            layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(self.input_dim, self.inner_dim)), nn.ReLU()))
            self.input_dim = self.inner_dim
        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(nn.Linear(self.inner_dim, output_dim), nn.Tanh())
        self.skip_layer = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim - self.skip_tensor_dim), nn.ReLU())

    def add_layer(self, device):
        new_layer = nn.Sequential(nn.utils.weight_norm(nn.Linear(self.input_dim, self.inner_dim)), nn.ReLU())
        new_layer.float().to(device)

        self.net.append(new_layer)
        self.num_layers += 1

    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor of shape (batch_size, 131). It contains a stacked tensor [latent_code, samples].
        Returns:
            sdf: output tensor of shape (batch_size, 1)
        """
        input_data = x.clone().detach()

        # Forward pass
        if self.skip_connections and self.num_layers >= 5:
            for i in range(3):
                x = self.net[i](x)
            x = self.skip_layer(x)
            x = torch.hstack((x, input_data))
            for i in range(self.num_layers - 5 - 1):  # last layer is residual
                x = self.net[3 + i](x)
            x = (1 - self.alpha) * x + self.alpha * self.net[-1](x)
            sdf = self.final_layer(x)
        else:
            if self.skip_connections:
                print('The network requires at least 5 layers to skip connections. Normal forward pass is used.')
            x = self.net(x)
            sdf = self.final_layer(x)
        return sdf


    def infer_latent_code(self, cfg, pointcloud, sdf_gt, writer, latent_code_initial):
        """Infer latent code from coordinates, their sdf, and a trained model."""

        latent_code = latent_code_initial.clone().detach().requires_grad_(True)

        optim = torch.optim.Adam([latent_code], lr=cfg['lr'])

        if cfg['lr_scheduler']:
            scheduler_latent = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                    factor=cfg['lr_multiplier'],
                                                    patience=cfg['patience'],
                                                    threshold=0.001, threshold_mode='rel')

        best_loss = 1000000

        for epoch in tqdm(range(0, cfg['epochs'])):

            latent_code_tile = torch.tile(latent_code, (pointcloud.shape[0], 1))
            x = torch.hstack((latent_code_tile, pointcloud))

            optim.zero_grad()

            predictions = self(x)

            if cfg['clamp']:
                predictions = torch.clamp(predictions, -cfg['clamp_value'], cfg['clamp_value'])

            loss_value, l1, l2 = utils_deepsdf.SDFLoss_multishape(sdf_gt, predictions, x[:, :self.latent_size], sigma=cfg['sigma_regulariser'], epsilon=0, lambdaa=0)
            loss_value.backward()

            if writer is not None:
                writer.add_scalar('Reconstruction loss', l1.data.cpu().numpy(), epoch)
                writer.add_scalar('Latent code loss', l2.data.cpu().numpy(), epoch)

            optim.step()

            if l1.detach().cpu().item() < best_loss:
                best_loss = l1.detach().cpu().item()
                best_latent_code = latent_code.clone()

            # step scheduler and store on tensorboard (optional)
            if cfg['lr_scheduler']:
                scheduler_latent.step(loss_value.item())
                if writer is not None:
                    writer.add_scalar('Learning rate', scheduler_latent._last_lr[0], epoch)

                if scheduler_latent._last_lr[0] < 1e-6:
                    print('Learning rate too small, stopping training')
                    break

            # logging
            if writer is not None:
                writer.add_scalar('Inference loss', loss_value.detach().cpu().item(), epoch)

        return best_latent_code

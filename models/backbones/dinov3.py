import torch
import torch.nn as nn

DINOV3_ARCHS = {
    'dinov3_vits16': 384,
    'dinov3_vitb16': 768,
    'dinov3_vitl16': 1024,
    'dinov3_vitg16': 1536,
}

class DINOv3(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,model_name='dinov3_vitb16',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
        ):
        super().__init__()

        assert model_name in DINOV3_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load(
                    repo_or_dir="/home/weitong/code/dinov3/",
                    model=model_name,
                    source='local',
                    pretrained=False
                )
        state_dict = torch.load("/home/weitong/code/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", map_location="cpu")
        self.model.load_state_dict(state_dict)

        self.model.to("cuda")
        self.num_channels = DINOV3_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token


    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 16, W // 16, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f

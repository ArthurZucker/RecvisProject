import torch.nn as nn
import torch


class Baseline(nn.Module):
    """
    Simple baseline head for semantic segmentation
    """

    def __init__(self, head_params, input_dim, img_size,patch_size = None):
        super().__init__()
        # just a linear layer of the flattened output of the embeddings
        # designed to use ViT outputs but should also work with resnet50
        # if input is [B,C,Patch], flatten to [B,CxPatch]
        self.proj = nn.Linear(input_dim, head_params.decoder_hidden_size)

        self.height = img_size[0]
        self.width  = img_size[1]
        self.patch_size = patch_size
        self.classifier = nn.Conv2d(head_params.decoder_hidden_size, head_params.num_labels, kernel_size=1)
        
    def forward(self, hidden_states: torch.Tensor):
        batch_size = hidden_states.shape[0]
        hidden_states = torch.flatten(hidden_states, start_dim=1) # ignore batch
        hidden_states = self.proj(hidden_states)
        encoder_hidden_state = hidden_states.permute(0, 2, 1)
        encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, self.height,self.width)
        # upsample
        encoder_hidden_state = nn.functional.interpolate(
            encoder_hidden_state, size=encoder_hidden_state.size()[2:], mode="bilinear", align_corners=False
        )
        logits = self.classifier(encoder_hidden_state)
        return logits

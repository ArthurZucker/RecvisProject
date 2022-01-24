# import torch.nn as nn
# import torch
# """
# HUGGING FACE https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/segformer/modeling_segformer.py#L697
# """

# class SegformerMLP(nn.Module):
#     """
#     Linear Embedding.
#     """

    # def __init__(self, config, input_dim):
    #     super().__init__()
    #     self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

#     def forward(self, hidden_states: torch.Tensor):
#         hidden_states = hidden_states.flatten(2).transpose(1, 2)
#         hidden_states = self.proj(hidden_states)
#         return hidden_states

# class SegformerDecodeHead(nn.Module): # SegformerPreTrainedModel
#     def __init__(self, config):
#         super().__init__()
#         # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
#         mlps = []
#         for i in range(config.num_encoder_blocks):
#             mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
#             mlps.append(mlp)
#         self.linear_c = nn.ModuleList(mlps)

#         # the following 3 layers implement the ConvModule of the original implementation
#         self.linear_fuse = nn.Conv2d(
#             in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
#             out_channels=config.decoder_hidden_size,
#             kernel_size=1,
#             bias=False,
#         )
#         self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
#         self.activation = nn.ReLU()

#         self.dropout = nn.Dropout(config.classifier_dropout_prob)
#         self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

#     def forward(self, encoder_hidden_states):
#         batch_size, _, _, _ = encoder_hidden_states[-1].shape
#         all_hidden_states = ()
#         for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
#             # unify channel dimension
#             height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
#             encoder_hidden_state = mlp(encoder_hidden_state)
#             encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
#             encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
#             # upsample
#             encoder_hidden_state = nn.functional.interpolate(
#                 encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
#             )
#             all_hidden_states += (encoder_hidden_state,)

#         hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
#         hidden_states = self.batch_norm(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         hidden_states = self.dropout(hidden_states)

#         # logits are of shape (batch_size, num_labels, height/4, width/4)
#         logits = self.classifier(hidden_states)

#         return logits
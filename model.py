import torch
from torch import nn
from x_transformers import TransformerWrapper, Encoder


class IC50Bert(nn.Module):
    def __init__(
        self,
        num_tokens: int = 256,
        max_seq_len: int = 2048,
        num_type_ids: int = 2,
        dim: int = 512,
        depth: int = 2,
        heads: int = 16,
        attn_kv_heads: int = 2,  # say you want 8 query heads to attend to 1 key / value head
        emb_dropout: float = 0.1,  # dropout after embedding
        layer_dropout: float = 0.1,  # stochastic depth - dropout entire layer
        attn_dropout: float = 0.1,  # dropout post-attention
        ff_dropout: float = 0.1,  # feedforward dropout
        attn_flash: bool = True,  # just set this to True if you have pytorch 2.0 installed
        use_rmsnorm: bool = True,  # set to true to use for all layers
        ff_swish: bool = True,  # set this to True
        ff_glu: bool = True,  # set to true to use for all feedforwards
        rotary_pos_emb: bool = True,
    ) -> None:
        super(IC50Bert, self).__init__()
        self.model = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            emb_dropout=emb_dropout,  # dropout after embedding
            embed_num_tokens={"token_type_ids": num_type_ids},
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                heads=heads,
                layer_dropout=layer_dropout,  # stochastic depth - dropout entire layer
                attn_dropout=attn_dropout,  # dropout post-attention
                ff_dropout=ff_dropout,  # feedforward dropout
                attn_flash=attn_flash,  # just set this to True if you have pytorch 2.0 installed
                use_rmsnorm=use_rmsnorm,  # set to true to use for all layers
                ff_swish=ff_swish,  # set this to True
                ff_glu=ff_glu,  # set to true to use for all feedforwards
                attn_kv_heads=attn_kv_heads,  # say you want 8 query heads to attend to 1 key / value head
                rotary_pos_emb=rotary_pos_emb,
            ),
        )

        self.dropout = nn.Dropout(ff_dropout)
        self.classifier = nn.Linear(dim, 1)

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(
            ids, mask=mask, embed_ids=token_type_ids, return_embeddings=True
        )  # (B, L, D)

        # print(outputs.shape)
        pooled_outputs = outputs[:, 0]
        pooled_outputs = self.dropout(pooled_outputs)

        # print(pooled_outputs.shape)

        logits = self.classifier(pooled_outputs)

        return logits


if __name__ == "__main__":
    x = torch.randint(0, 256, (4, 2048))
    mask = torch.ones_like(x).bool()
    types = torch.randint(0, 2, (4, 2048))

    model = IC50Bert()

    out = model(
        x, mask=mask, token_type_ids={"token_type_ids": types}
    )  # (1, 1024, 200)
    print(out.shape)

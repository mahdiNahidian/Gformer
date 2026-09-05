import torch
import torch.nn as nn

from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attention import (
    FullAttention,
    GradualProbSparseAttention,
    AttentionLayer,
)
from models.embed import DataEmbedding


class Gformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 d_ff=512, dropout=0.0, attn='prob', embed='fixed', freq='h',
                 activation='gelu', output_attention=False, distil=True,
                 mix=True, device=torch.device('cuda:0')):
        super(Gformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout
        )
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout
        )

        def make_encoder_attention():
            if attn == 'custom':
                return GradualProbSparseAttention(
                    mask_flag=False,
                    factor=factor,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                    warmup_epochs=2,
                    removal_rate=0.1,
                )
            return FullAttention(
                mask_flag=False,
                factor=factor,
                attention_dropout=dropout,
                output_attention=output_attention,
            )

        def make_decoder_self_attention():
            if attn == 'custom':
                return GradualProbSparseAttention(
                    mask_flag=True,
                    factor=factor,
                    attention_dropout=dropout,
                    output_attention=False,
                    warmup_epochs=2,
                    removal_rate=0.1,
                )
            return FullAttention(
                mask_flag=True,
                factor=factor,
                attention_dropout=dropout,
                output_attention=False,
            )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        make_encoder_attention(),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            [
                ConvLayer(d_model)
                for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        make_decoder_self_attention(),
                        d_model,
                        n_heads,
                        mix=mix,
                    ),
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None,
                dec_enc_mask=None, epoch=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(
            enc_out,
            attn_mask=enc_self_mask,
            epoch=epoch,
        )

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            epoch=epoch,
        )
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]


class GformerStack(nn.Module):
    """
    Stacked encoder variant of Gformer.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1],
                 d_layers=2, d_ff=512, dropout=0.0, attn='prob',
                 embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(GformerStack, self).__init__()
        self.pred_len = out_len
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout
        )
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout
        )

        def make_encoder_attention():
            if attn == 'custom':
                return GradualProbSparseAttention(
                    mask_flag=False,
                    factor=factor,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                    warmup_epochs=2,
                    removal_rate=0.1,
                )
            return FullAttention(
                mask_flag=False,
                factor=factor,
                attention_dropout=dropout,
                output_attention=output_attention,
            )

        inp_lens = list(range(len(e_layers)))
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            make_encoder_attention(),
                            d_model,
                            n_heads,
                            mix=False,
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for _ in range(el)
                ],
                [
                    ConvLayer(d_model)
                    for _ in range(el - 1)
                ] if distil else None,
                norm_layer=nn.LayerNorm(d_model),
            )
            for el in e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)

        def make_decoder_self_attention():
            if attn == 'custom':
                return GradualProbSparseAttention(
                    mask_flag=True,
                    factor=factor,
                    attention_dropout=dropout,
                    output_attention=False,
                    warmup_epochs=2,
                    removal_rate=0.1,
                )
            return FullAttention(
                mask_flag=True,
                factor=factor,
                attention_dropout=dropout,
                output_attention=False,
            )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        make_decoder_self_attention(),
                        d_model,
                        n_heads,
                        mix=mix,
                    ),
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None,
                dec_enc_mask=None, epoch=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(
            enc_out,
            attn_mask=enc_self_mask,
            epoch=epoch,
        )

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            epoch=epoch,
        )
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]

"""
Qwen2 Decoder-as-Encoder (D2E) — Visual Causal Flow mechanism.

Uses a Qwen2-0.5B transformer as a visual encoder with a custom
attention mask that combines:
  - Non-causal (bidirectional) attention for image tokens
  - Causal attention for learnable query tokens that read from image tokens
"""

import torch
import torch.nn as nn
import transformers
from typing import Optional, Tuple


class CustomQwen2Decoder(nn.Module):
    """
    Qwen2 visual encoder with custom Visual Causal Flow attention mask.

    token_type_ids:
        0 = image token (non-causal, full bidirectional attention)
        1 = query token (causal, attends to all image + causally to prior queries)
    """

    def __init__(
        self,
        decoder_layer: int = 24,
        max_position_embeddings: int = 131072,
        hidden_dimension: int = 896,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
        vocab_size: int = 151936,
        attn_implementation: str = "sdpa",
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
    ):
        super().__init__()

        if attn_implementation == "flash_attention_2":
            raise ValueError(
                "CustomQwen2Decoder does not support flash_attention_2; "
                "the custom attention mask requires 'sdpa' or 'eager'."
            )

        Qwen2Model = getattr(transformers.models.qwen2.modeling_qwen2, 'Qwen2Model')
        Qwen2Config = getattr(transformers, 'Qwen2Config')

        config = Qwen2Config(
            hidden_size=hidden_dimension,
            num_hidden_layers=decoder_layer,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            _attn_implementation=attn_implementation,
        )

        self.model = self._create_custom_model(Qwen2Model, config)
        del self.model.embed_tokens

    def _create_custom_model(self, Qwen2Model, config):
        """Subclass Qwen2Model to inject the Visual Causal Flow attention mask."""

        class CustomQwen2ModelInner(Qwen2Model):

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                token_type_ids=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                cache_position=None,
            ):
                self._current_token_type_ids = token_type_ids

                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

            def _update_causal_mask(
                self,
                attention_mask,
                input_tensor,
                cache_position,
                past_key_values,
                output_attentions,
            ):
                dtype, device = input_tensor.dtype, input_tensor.device
                min_dtype = torch.finfo(dtype).min
                batch_size, sequence_length = input_tensor.shape[0], input_tensor.shape[1]

                causal_mask = self._create_custom_4d_mask(
                    sequence_length=sequence_length,
                    dtype=dtype,
                    device=device,
                    batch_size=batch_size,
                    token_type_ids=self._current_token_type_ids,
                )

                if attention_mask is not None and attention_mask.dim() == 2:
                    padding_mask = attention_mask[:, None, None, :].to(dtype=dtype)
                    padding_mask = (1.0 - padding_mask) * min_dtype
                    causal_mask = causal_mask + padding_mask

                return causal_mask

            def _create_custom_4d_mask(
                self,
                sequence_length,
                dtype,
                device,
                batch_size,
                token_type_ids,
            ):
                """Build the Visual Causal Flow 4-D attention mask.

                Pattern:
                  Image → Image : full bidirectional (non-causal)
                  Query → Image : full cross-attention
                  Query → Query : lower-triangular (causal)
                  Image → Query : blocked (image tokens cannot see future queries)
                """
                min_dtype = torch.finfo(dtype).min
                masks = []
                for b in range(batch_size):
                    mask = torch.full(
                        (sequence_length, sequence_length),
                        fill_value=min_dtype,
                        dtype=dtype,
                        device=device,
                    )
                    type_ids = token_type_ids[b]
                    image_positions = (type_ids == 0).nonzero(as_tuple=True)[0]
                    text_positions = (type_ids == 1).nonzero(as_tuple=True)[0]

                    if len(image_positions) > 0:
                        mask[image_positions[:, None], image_positions] = 0.0

                    for i, text_pos in enumerate(text_positions):
                        if len(image_positions) > 0:
                            mask[text_pos, image_positions] = 0.0
                        mask[text_pos, text_positions[:i + 1]] = 0.0

                    masks.append(mask)

                return torch.stack(masks, dim=0).unsqueeze(1)

        return CustomQwen2ModelInner(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            inputs_embeds: [B, seq_len, hidden_dim]
            token_type_ids: [B, seq_len], 0=image, 1=query
            attention_mask: [B, seq_len], optional padding mask

        Returns:
            BaseModelOutputWithPast
        """
        return self.model(
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            **kwargs,
        )


class Qwen2Decoder2Encoder(nn.Module):
    """
    Full Decoder-as-Encoder module.

    Takes SAM feature maps [B, 896, H, W], flattens them, concatenates
    learned query embeddings, runs through the D2E transformer with the
    Visual Causal Flow attention mask, and returns the query outputs
    [B, n_queries, 896].
    """

    def __init__(
        self,
        decoder_layer: int = 24,
        hidden_dimension: int = 896,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
        max_query: int = 400,
        attn_implementation: str = "sdpa",
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        super().__init__()

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        self.model = CustomQwen2Decoder(
            decoder_layer=decoder_layer,
            hidden_dimension=hidden_dimension,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            attn_implementation=attn_implementation,
        )

        # Learnable query embeddings (one set per resolution)
        self.query_768 = nn.Embedding(144, hidden_dimension)   # 12×12 queries for 768px input
        self.query_1024 = nn.Embedding(256, hidden_dimension)  # 16×16 queries for 1024px input

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: SAM encoder output [B, 896, H, W]
            output_attentions: override instance default
            output_hidden_states: override instance default

        Returns:
            If output_attentions/output_hidden_states are False:
                query_outputs: [B, n_queries, 896]
            Otherwise returns a tuple:
                (query_outputs, attentions, hidden_states, token_type_ids)
        """
        _out_attn = output_attentions if output_attentions is not None else self.output_attentions
        _out_hs = output_hidden_states if output_hidden_states is not None else self.output_hidden_states

        x = x.flatten(2).transpose(1, 2)  # [B, H*W, 896]
        bs, n_img, _ = x.shape

        if n_img == 144:
            param_img = self.query_768.weight
        elif n_img == 256:
            param_img = self.query_1024.weight
        else:
            raise ValueError(f"Unexpected spatial token count: {n_img}. Expected 144 (768px) or 256 (1024px).")

        batch_queries = param_img.unsqueeze(0).expand(bs, -1, -1)
        x_combined = torch.cat([x, batch_queries], dim=1)

        token_type_ids = torch.cat([
            torch.zeros(bs, n_img, dtype=torch.long, device=x.device),
            torch.ones(bs, n_img, dtype=torch.long, device=x.device),
        ], dim=1)

        outputs = self.model(
            x_combined,
            token_type_ids,
            output_attentions=_out_attn,
            output_hidden_states=_out_hs,
        )

        y = outputs[0][:, n_img:, :]  # Extract causal query outputs

        if _out_attn or _out_hs:
            attentions = outputs.attentions if _out_attn else None
            hidden_states = outputs.hidden_states if _out_hs else None
            return y, attentions, hidden_states, token_type_ids

        return y


def build_qwen2_decoder_as_encoder(
    decoder_layer: int = 24,
    hidden_dimension: int = 896,
    num_attention_heads: int = 14,
    num_key_value_heads: int = 2,
    intermediate_size: int = 4864,
    max_query: int = 400,
    attn_implementation: str = "sdpa",
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    checkpoint: Optional[str] = None,
) -> Qwen2Decoder2Encoder:
    """Build and optionally load a Qwen2 Decoder-as-Encoder."""
    decoder_as_encoder = Qwen2Decoder2Encoder(
        decoder_layer=decoder_layer,
        hidden_dimension=hidden_dimension,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_query=max_query,
        attn_implementation=attn_implementation,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu")
        decoder_as_encoder.load_state_dict(state_dict, strict=True)
        print(f"Loaded D2E checkpoint: {checkpoint}")

    return decoder_as_encoder

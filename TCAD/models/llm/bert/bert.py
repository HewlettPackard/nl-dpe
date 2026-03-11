###
# Copyright (2026) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

'''Modified from Transformer version v5.0.0.
'''
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import initialization as init
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig

from models.utils.layers import Matmul, Softmax


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids = None,
        token_type_ids = None,
        position_ids = None,
        inputs_embeds = None,
        past_key_values_length = 0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.layer_idx = layer_idx

        self.matmul1 = Matmul()
        self.matmul2 = Matmul()
        self.softmax = Softmax()

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        cache_position = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

        attn_weights = self.matmul1(query_layer, key_layer.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_layer.shape[-2]]
            attn_weights = attn_weights + attention_mask

        attn_weights = self.softmax(attn_weights, dim=-1)

        dropout = 0.0 if not self.training else self.dropout.p
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)

        attn_output = self.matmul2(attn_weights, value_layer)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.self = BertSelfAttention(config, layer_idx=layer_idx)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        cache_position = None,
        **kwargs,
    ):
        attention_output, attn_weights = self.self(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = self.output(attention_output, hidden_states)
        return attention_output, attn_weights


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.attention = BertAttention(config, layer_idx=layer_idx)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        cache_position = None,
        **kwargs,
    ):
        self_attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = self_attention_output

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        cache_position = None,
        **kwargs,
    ):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                cache_position=cache_position,
                **kwargs,
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": BertLayer,
        "attentions": BertSelfAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, BertEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
            init.zeros_(module.token_type_ids)


class BertModel(BertPreTrainedModel):
    _no_split_modules = ["BertEmbeddings", "BertLayer"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        inputs_embeds = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        past_key_values = None,
        use_cache = None,
        cache_position = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            device = input_ids.device
            seq_length = input_ids.shape[1]
        else:
            device = inputs_embeds.device
            seq_length = inputs_embeds.shape[1]

        past_key_values_length = 0
        if cache_position is None:
            cache_position = torch.arange(past_key_values_length, past_key_values_length + seq_length, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        attention_mask, encoder_attention_mask = self._create_attention_masks(
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            embedding_output=embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            cache_position=cache_position,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )

    def _create_attention_masks(
        self,
        attention_mask,
        encoder_attention_mask,
        embedding_output,
        encoder_hidden_states,
        cache_position,
    ):
        attention_mask = create_bidirectional_mask(
            config=self.config,
            input_embeds=embedding_output,
            attention_mask=attention_mask,
        )

        if encoder_attention_mask is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        return attention_mask, encoder_attention_mask


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        inputs_embeds = None,
        labels = None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        inputs_embeds = None,
        start_positions = None,
        end_positions = None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

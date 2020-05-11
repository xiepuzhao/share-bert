
# todo:动态调整candidate set的大小
# todo:搞清楚seen和unseen之间的关系，从集合角度
# todo:考虑加入mention string和entity title,在下一个branch里面实现验证

from transformers import BertPreTrainedModel, BertModel, BertTokenizer, BertConfig
from torch import nn
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
import torch
from torch.nn import CrossEntropyLoss
from bert_model import BertSpecificModel,PreTrainedSpecificModel,BertSpecificPreTrainedModel

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""

BERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""
# 要注意模型在处理recall,rank阶段时num_lables的值是不同的
# config.num_labels的值为entity的数量


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class EntityRecallRank(BertSpecificPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 这里num_labels的数量应为entity_set中实体的数量
        self.num_labels = config.num_labels
        # self.bert = BertModel(config)
        self.bert = BertSpecificModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.recall_classifier = nn.Linear(config.hidden_size, 104520)
        # 利用从文件读入的entity_embedding对recall_classifier的权重进行初始化
        # self.recall_classifier.weight = nn.Parameter(entity_embedding)
        self.rank_classifier = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            rank=False,
            recall=False,
    ):
        assert (rank ^ recall) is True
        # todo:similar to fast-bert,add 12 probes, find which is suitable
        if recall:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,  # shape(104520, 768)
                recall=True,
                rank=False,
            )
            # Sequence of hidden-states at the output of the last layer of the model
            # shape:(batch_size, sequence_length, hidden_size)
            # 要考虑padding的可能，即padding的token embedding不能参与到mean pooling中
            last_hidden_state = outputs[0]
            last_hidden_state = self.dropout(last_hidden_state)
            # unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度
            # expand函数将向量的单个维度扩展为更大的尺寸
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

            # todo:看看torch.clamp是否有作用
            sum_token_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            token_num = input_mask_expanded.sum(1)
            # token_num = torch.clamp(token_num, min=1e-9)
            mean_token_embeddings = sum_token_embeddings/token_num
            logits = self.recall_classifier(mean_token_embeddings)
            outputs = (logits,) + outputs[2:]

            if labels is not None:
                loss_func = CrossEntropyLoss()
                loss = loss_func(logits.view(-1, 104520), labels.view(-1))
                outputs = (loss,) + outputs
        if rank:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,  # shape(104520, 768)
                recall=False,
                rank=True,
            )

            # shape:(batch_size, hidden_size)
            last_hidden_state = outputs[0]
            last_hidden_state = self.dropout(last_hidden_state)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

            sum_token_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_tokens = input_mask_expanded.sum(1)
            sum_tokens = torch.clamp(sum_tokens, min=1e-9)
            mean_token_embeddings = sum_token_embeddings / sum_tokens
            logits = self.rank_classifier(mean_token_embeddings)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs

# bert_dir = '/home/puzhao_xie/entity-linking-task/candidate_generation/sentence_transformers/output/training_nli_/home/puzhao_xie/.cache/torch/sentence_transformers/public.ukp.informatik.tu-darmstadt.de_reimers_sentence-transformers_v0.2_bert-base-nli-stsb-mean-tokens.zip/0_BERT-2020-03-30_23-19-16/0_BERT'
# tokenizer = BertTokenizer.from_pretrained(bert_dir)
# config = BertConfig.from_pretrained(
#         bert_dir,
#         output_hidden_states=True,
#         num_hidden_layers=11,
#     )
# bertSpecificModel = EntityRecallRank(config=config)
# # bertModel = BertModel(config=config)
# # bertModel = bertModel.from_pretrained(pretrained_model_name_or_path=bert_dir, config=config)
# bertSpecificModel = bertSpecificModel.from_pretrained(pretrained_model_name_or_path=bert_dir, config=config, state_dict=bertSpecificModel.state_dict())
# # print(bertSpecificModel.state_dict()[])
# for key in bertSpecificModel.named_parameters():
#     print(key)










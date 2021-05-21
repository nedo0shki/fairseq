# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding, TransformerDecoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import new_arange

from .levenshtein_utils import (
    _apply_del_words,
    _apply_ins_masks,
    _apply_ins_words,
    _fill,
    _get_del_targets,
    _get_ins_targets,
    _skip,
    _skip_encoder_out,
)


def _random_delete(target_tokens, bos, eos, pad):

    max_len = target_tokens.size(1)
    target_mask = target_tokens.eq(pad)
    target_score = target_tokens.clone().float().uniform_()
    target_score.masked_fill_(
        target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
    )
    target_score.masked_fill_(target_mask, 1)
    target_score, target_rank = target_score.sort(1)
    target_length = target_mask.size(1) - target_mask.float().sum(
        1, keepdim=True
    )

    # do not delete <bos> and <eos> (we assign 0 score for them)
    target_cutoff = (
        2
        + (
            (target_length - 2)
            * target_score.new_zeros(target_score.size(0), 1).uniform_()
        ).long()
    )
    target_cutoff = target_score.sort(1)[1] >= target_cutoff

    prev_target_tokens = (
        target_tokens.gather(1, target_rank)
        .masked_fill_(target_cutoff, pad)
        .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
    )
    '''
    prev_target_tokens = prev_target_tokens[
        :, : prev_target_tokens.ne(pad).sum(1).max()
    ]
    '''
    return prev_target_tokens

def _expert_delete(prev_output_tokens, tgt_tokens, pad):

    word_del_targets = _get_del_targets(prev_output_tokens, tgt_tokens, pad)
    max_len = prev_output_tokens.size(1)
    reordering = new_arange(prev_output_tokens).masked_fill_(word_del_targets.bool(), max_len).sort(1)[1]
    del_tokens = prev_output_tokens.masked_fill(word_del_targets.bool(), pad).gather(1, reordering)
    return del_tokens





@register_model("levenshtein_transformer")
class LevenshteinTransformerModel(FairseqNATModel):
    @property
    def allow_length_beam(self):
        return False

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        parser.add_argument(
            "--early-exit",
            default="6,6,6",
            type=str,
            help="number of decoder layers before word_del, mask_ins, word_ins",
        )

        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )

        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )

        parser.add_argument(
            "--share-discriminator-maskpredictor",
            action="store_true",
            help="share the parameters for both mask-predictor and discriminator",
        )

        parser.add_argument(
            "--sampling-for-deletion",
            action="store_true",
            help="instead of argmax, use sampling to predict the tokens",
        )

        parser.add_argument(
            "--dae-ratio",
            type=float,
            help="the probability of using noise injected target to learn the insertion policy",
        )

        parser.add_argument(
            "--alpha-ratio",
            type=float,
            help="the probability of using init tokens to learn the deletion policy",
        )

        parser.add_argument(
            "--within-batch", action="store_true",
            help="mixed dae and actual distribution within batch",
        )

        parser.add_argument(
            "--train-step",
            type=int,
            help="number of refinement iterations during training",
        )

        parser.add_argument(
            "--dae-for-first-iter", action="store_true",
            help="use dae just in the first step of training steps",
        )

        parser.add_argument(
            "--roll-out", action="store_true",
            help="aggregate roll-out data during training",
        )

        parser.add_argument(
            "--model-del", action="store_true",
            help="use model delete instead of the oracle to learn the insertion policy",
        )


    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        model.dae_ratio = getattr(args, "dae_ratio", 0.5)
        model.alpha_ratio = getattr(args, "alpha_ratio", 0.5)
        model.within_batch = getattr(args, "within_batch", False)
        model.train_step = getattr(args, "train_step", 1)
        model.dae_for_first_iter = getattr(args, "dae_for_first_iter", False)
        model.random_policy = getattr(args, "random_policy", "rnd-del")
        model.model_del = getattr(args, "model_del", False)
        model.roll_out = getattr(args, "roll_out", False)
        return model

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LevenshteinTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, train_step, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        B, T = tgt_tokens.size()
        V = len(self.decoder.dictionary)

        if train_step is None:
            train_step = 1

        if self.roll_out:
            if train_step > 1:
                init_tokens_list = []
                init_tokens_list.append(prev_output_tokens)
                for t in range(train_step - 1):
                    output_scores = init_tokens_list[t].new_zeros(
                        *init_tokens_list[t].size()
                    ).type_as(encoder_out["encoder_out"][0])
                    decoder_out = DecoderOut(
                        output_tokens=init_tokens_list[t],
                        output_scores=output_scores,
                        attn=None,
                        step=0,
                        max_step=0,
                        history=None
                    )
                    decoder_out = self.forward_decoder(decoder_out, encoder_out, None, max_ratio=2)
                    init_tokens_list.append(decoder_out.output_tokens)
                inits_max_len = max([init_tokens.size(1) for init_tokens in init_tokens_list])
                print(inits_max_len)
                for i in range(len(init_tokens_list)):
                    print("before: ",init_tokens_list[i].size())
                    if init_tokens_list[i].size(1) < inits_max_len:
                        pads = init_tokens_list[i].new_full((init_tokens_list[i].size(0),inits_max_len - init_tokens_list[i].size(1)),self.pad)
                        init_tokens_list[i] = torch.cat([init_tokens_list[i],pads],1)
                    print("after:", init_tokens_list[i].size())
                prev_output_tokens = init_tokens_list[0]
                rand_ind = torch.rand(size=(B,), device=tgt_tokens.device)
                for t in range(train_step):
                    curr_ind = (rand_ind > t/train_step) & (rand_ind <= (t+1)/train_step)
                    prev_output_tokens[curr_ind] = init_tokens_list[t][curr_ind]

        if prev_output_tokens.size(1) > tgt_tokens.size(1):
            pads = tgt_tokens.new_full((tgt_tokens.size(0),prev_output_tokens.size(1) - tgt_tokens.size(1)),self.pad)
            tgt_tokens = torch.cat([tgt_tokens,pads],1)
        if prev_output_tokens.size(1) < tgt_tokens.size(1):
            pads = prev_output_tokens.new_full((prev_output_tokens.size(0), tgt_tokens.size(1) - prev_output_tokens.size(1)),self.pad)
            prev_output_tokens = torch.cat([prev_output_tokens,pads],1)


        if self.within_batch:
            corrupted = (
                            torch.rand(size=(B,), device=tgt_tokens.device)
                            < self.dae_ratio
                        )
            if self.random_policy == "rnd-del":
                noisy_target = _random_delete(tgt_tokens, self.bos, self.eos, self.pad)
            else:
                noisy_target = _sequential_poisoning(prev_output_tokens, V, 0.33, self.bos, self.eos, self.pad)
                noisy_target = _expert_delete(noisy_target, tgt_tokens, self.pad)

            if self.model_del:

                word_del_score, word_del_attn = self.decoder.forward_word_del(
                    normalize=True,
                    prev_output_tokens=prev_output_tokens,
                    encoder_out=encoder_out,
                )

                word_del_pred = word_del_score.max(-1)[1].bool()

                y_ins, _, _ = _apply_del_words(
                    prev_output_tokens,
                    None,
                    None,
                    word_del_pred,
                    self.pad,
                    self.bos,
                    self.eos,
                )
            else:
                y_ins = _expert_delete(prev_output_tokens, tgt_tokens, self.pad)

            y_ins[corrupted] = noisy_target[corrupted]

        else:
            if random.uniform(0,1) < self.dae_ratio:
                y_ins = _random_delete(tgt_tokens, self.bos, self.eos, self.pad)
            else:
                y_ins = _expert_delete(prev_output_tokens, tgt_tokens, self.pad)

        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_target = _get_ins_targets(
            y_ins, tgt_tokens, self.pad, self.unk
        )
        mask_ins_target = mask_ins_target.clamp(min=0, max=255)  # for safe prediction
        mask_ins_mask = y_ins[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False,
            prev_output_tokens=y_ins,
            encoder_out=encoder_out,
        )


        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            prev_output_tokens=masked_tgt_tokens,
            encoder_out=encoder_out,
        )

        # make online prediction
        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1
            ).view(word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(
            ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        )

        # generate training labels for deletion
        if self.within_batch:
            y_del = word_predictions
            init_seq = (
                            torch.rand(size=(B,), device=tgt_tokens.device)
                            < self.alpha_ratio
                        )
            y_del[init_seq] = prev_output_tokens[init_seq]
        else:
            if prev_output_tokens.size(1) > 2:
                if random.uniform(0,1) < self.alpha_ratio:
                    y_del = prev_output_tokens
                else:
                    y_del = word_predictions
            else:
                y_del = word_predictions



        word_del_target = _get_del_targets(y_del, tgt_tokens, self.pad)
        word_del_out, _ = self.decoder.forward_word_del(
            normalize=False,
            prev_output_tokens=y_del,
            encoder_out=encoder_out,
        )
        word_del_mask = y_del.ne(self.pad)

        return {
            "mask_ins": {
                "out": mask_ins_out,
                "tgt": mask_ins_target,
                "mask": mask_ins_mask,
                "ls": 0.01,
            },
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": masked_tgt_masks,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "word_del": {
                "out": word_del_out,
                "tgt": word_del_target,
                "mask": word_del_mask,
            },

        }


    def forward_decoder(
        self, decoder_out, encoder_out, target_tokens, eos_penalty=0.0, max_ratio=None,
        oracle_del=False, oracle_ins=False, **kwargs
    ):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history

        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if not encoder_out["encoder_padding_mask"]:
                max_src_len = encoder_out["encoder_out"].size(0)
                src_lens = encoder_out["encoder_out"].new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out["encoder_padding_mask"][0]).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()
        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            if oracle_del:
                word_del_pred = _get_del_targets(
                    _skip(output_tokens, can_del_word),
                    _skip(target_tokens, can_del_word), self.pad).bool()
                word_del_attn = None
                word_del_score = None
            else:
                word_del_score, word_del_attn = self.decoder.forward_word_del(
                    normalize=True,
                    prev_output_tokens=_skip(output_tokens, can_del_word),
                    encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_del_word),
                )
                word_del_pred = word_del_score.max(-1)[1].bool()

            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                word_del_attn,
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            attn = _fill(attn, can_del_word, _attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_mask),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_mask),
            )
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
            )

            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone())

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_word),
            )
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, init_tokens=None):
        if init_tokens is not None:
            initial_output_tokens = init_tokens
        else:
            initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens[:, 1] = self.eos

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )


class LevenshteinTransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        #changed 256 to 100 however it seems that there was a sample needed more than that
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(",")]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_msk = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[1])
                ]
            )
        self.layers_del = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_del = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[0])
                ]
            )

        if getattr(args, "share_discriminator_maskpredictor", False):
            assert getattr(
                args, "no_share_discriminator", False
            ), "must set saperate discriminator"
            self.layers_msk = self.layers_del

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        layers=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[:early_exit]):
            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    @ensemble_decoder
    def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[1],
            layers=self.layers_msk,
            **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    @ensemble_decoder
    def forward_word_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[2],
            layers=self.layers,
            **unused
        )
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    @ensemble_decoder
    def forward_word_del(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[0],
            layers=self.layers_del,
            **unused
        )
        decoder_out = F.linear(features, self.embed_word_del.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]


@register_model_architecture("levenshtein_transformer", "levenshtein_transformer")
def levenshtein_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(
        args, "share_discriminator_maskpredictor", False
    )
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)


@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_wmt_en_de"
)
def levenshtein_transformer_wmt_en_de(args):
    levenshtein_base_architecture(args)


# similar parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_vaswani_wmt_en_de_big"
)
def levenshtein_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    levenshtein_base_architecture(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_wmt_en_de_big"
)
def levenshtein_transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    levenshtein_transformer_vaswani_wmt_en_de_big(args)

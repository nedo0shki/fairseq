# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder, LevenshteinTransformerDecoder
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

def summary(predictions, ground_truths):
    summary_df = []
    for prediction, ground_truth in zip(predictions, ground_truths):
        false_pos = 0
        true_pos = 0
        false_neg = 0
        true_neg = 0
        for p,t in zip(prediction, ground_truth):
            if p == True:
                if t == True:
                    true_pos = true_pos + 1
                else:
                    false_pos = false_pos + 1
            else:
                if t == True:
                    false_neg = false_neg + 1
                else:
                    true_neg = true_neg + 1
        summary_df.append({"true_pos": true_pos, "false_pos": false_pos, "true_neg": true_neg, "false_neg": false_neg})
    summary_df = pd.DataFrame(summary_df)
    return(summary_df.sum())

def _random_delete(target_tokens, bos, eos, pad, rnd_type):

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

    rnd_score = target_score.new_zeros(target_score.size(0), 1).uniform_()

    if rnd_type == 'exp':
        rnd_score = (1 - torch.exp(-2*rnd_score))/(1 - torch.exp(torch.tensor([-2.0]).to(rnd_score.device)))
    # do not delete <bos> and <eos> (we assign 0 score for them)
    target_cutoff = (
        2
        + (
            (target_length - 2)
            * rnd_score
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



@register_model("iterative_levenshtein_transformer")
class IterativeLevenshteinTransformerModel(FairseqNATModel):
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
            "--factor-ratio",
            type=float,
            help="ratio of factor decreasing for further steps",
        )


        parser.add_argument(
            "--train-step",
            type=int,
            help="number of refinement iterations during training",
        )




    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)


    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        model.train_step = getattr(args, "train_step", 1)
        model.factor_ratio = getattr(args, "factor_ratio", 1)
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


        if prev_output_tokens.size(1) > tgt_tokens.size(1):
            pads = tgt_tokens.new_full((tgt_tokens.size(0),prev_output_tokens.size(1) - tgt_tokens.size(1)),self.pad)
            tgt_tokens = torch.cat([tgt_tokens,pads],1)
        if prev_output_tokens.size(1) < tgt_tokens.size(1):
            pads = prev_output_tokens.new_full((prev_output_tokens.size(0), tgt_tokens.size(1) - prev_output_tokens.size(1)),self.pad)
            prev_output_tokens = torch.cat([prev_output_tokens,pads],1)

        objs = {}
        curr_factor = 1
        for step in range(self.train_step):

            word_del_target = _get_del_targets(prev_output_tokens, tgt_tokens, self.pad)
            word_del_out, _ = self.decoder.forward_word_del(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )
            word_del_mask = prev_output_tokens.ne(self.pad)

            objs["word_del"+str(step)] = {
                "out": word_del_out,
                "tgt": word_del_target,
                "mask": word_del_mask,
                "factor": curr_factor,
            }

            word_del_pred = word_del_out.max(-1)[1].bool()
            prev_output_tokens, _, _ = _apply_del_words(
                prev_output_tokens,
                None,
                None,
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )

            masked_tgt_masks, masked_tgt_tokens, mask_ins_target = _get_ins_targets(
                prev_output_tokens, tgt_tokens, self.pad, self.unk
            )
            mask_ins_target = mask_ins_target.clamp(min=0, max=255)  # for safe prediction
            mask_ins_mask = prev_output_tokens[:, 1:].ne(self.pad)

            mask_ins_out, _ = self.decoder.forward_mask_ins(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )


            word_ins_out, _ = self.decoder.forward_word_ins(
                normalize=False,
                prev_output_tokens=masked_tgt_tokens,
                encoder_out=encoder_out,
            )

            objs["mask_ins"+str(step)] = {
                "out": mask_ins_out,
                "tgt": mask_ins_target,
                "mask": mask_ins_mask,
                "ls": 0.01,
                "factor": curr_factor,
            }

            objs["word_ins"+str(step)] = {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": masked_tgt_masks,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": curr_factor,
            }

            if self.decoder.sampling_for_deletion:
                word_predictions = torch.multinomial(
                    F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1
                ).view(word_ins_out.size(0), -1)
            else:
                word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

            word_predictions.masked_scatter_(
                ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
            )

            prev_output_tokens = word_predictions

            curr_factor = curr_factor * self.factor_ratio

        return objs


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

                '''
                s = summary(word_del_pred,word_del_pred)
                recall = s['true_pos'] / (s['true_pos'] + s['false_neg'])
                precision = s['true_pos'] / (s['true_pos'] + s['false_pos'])
                print(_skip(output_tokens, can_del_word).size(0), "sentences in step ", decoder_out.step)
                print(s['true_pos'] + s['false_neg'], " should be deleted")
                print(s['true_pos'] + s['false_pos'], " deleted")
                print("recall: ", recall, ", precision: ", precision)
                '''
            else:

                if decoder_out.step == 0:

                    word_del_score, word_del_attn = self.decoder.first_forward_word_del(
                        normalize=True,
                        prev_output_tokens=_skip(output_tokens, can_del_word),
                        encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_del_word),
                    )
                    word_del_pred = word_del_score.max(-1)[1].bool()

                    '''
                    oracle_word_del_pred = _get_del_targets(
                        _skip(output_tokens, can_del_word),
                        _skip(target_tokens, can_del_word), self.pad).bool()
                    s = summary(word_del_pred,oracle_word_del_pred)
                    recall = s['true_pos'] / (s['true_pos'] + s['false_neg'])
                    precision = s['true_pos'] / (s['true_pos'] + s['false_pos'])
                    print(_skip(output_tokens, can_del_word).size(0), "sentences in step ", decoder_out.step)
                    print(s['true_pos'] + s['false_neg'], " should be deleted")
                    print(s['true_pos'] + s['false_pos'], " deleted")
                    print("recall: ", recall, ", precision: ", precision)
                    '''
                else:
                    word_del_score, word_del_attn = self.decoder.forward_word_del(
                        normalize=True,
                        prev_output_tokens=_skip(output_tokens, can_del_word),
                        encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_del_word),
                    )
                    word_del_pred = word_del_score.max(-1)[1].bool()

                    '''
                    oracle_word_del_pred = _get_del_targets(
                        _skip(output_tokens, can_del_word),
                        _skip(target_tokens, can_del_word), self.pad).bool()
                    s = summary(word_del_pred,oracle_word_del_pred)
                    recall = s['true_pos'] / (s['true_pos'] + s['false_neg'])
                    precision = s['true_pos'] / (s['true_pos'] + s['false_pos'])
                    print(_skip(output_tokens, can_del_word).size(0), "sentences in step ", decoder_out.step)
                    print(s['true_pos'] + s['false_neg'], " should be deleted")
                    print(s['true_pos'] + s['false_pos'], " deleted")
                    print("recall: ", recall, ", precision: ", precision)
                    '''

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
            print("mask pred at step ", decoder_out.step, ": ", torch.sum(mask_ins_pred,1))
            print("total mask pred at step ", decoder_out.step, ": ", torch.sum(mask_ins_pred))
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




@register_model_architecture("iterative_levenshtein_transformer", "iterative_levenshtein_transformer")
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

import heapq
import logging
import math
from collections import Counter, defaultdict

from scipy.special import digamma
from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import Pretokenizer
from script_bpe.unigram.model import UnigramToken, UnigramModel
from script_bpe.utils import token_array, create_logger

MIN_EXPECTED_COUNT = 0.01  # in m-step, for avoiding underflows


def log_examples(
    logger,
    pretokenizer: Pretokenizer,
    tokens_with_scores: list[tuple[UnigramToken, float]],
    score_label="score",
    n=5,
):
    tokens_with_scores.sort(key=lambda x: x[1])
    for i, (t, score) in enumerate(tokens_with_scores):
        if len(tokens_with_scores) > 2 * n:
            if i == n:
                logger.debug("   │  ├─ ...")
            if n < i < len(tokens_with_scores) - n:
                continue
        list_item = " ├─" if i < len(tokens_with_scores) - 1 else " └─"
        logger.debug(
            f"   │ {list_item} {repr(pretokenizer.decode(t.base_tokens)):25}  {score_label} = {score:10.3g}  base_tokens = {list(t.base_tokens)}"
        )


def make_initial_vocab(
    logger: logging.Logger,
    corpus: PretokenizedCorpus,
    pretokenizer: Pretokenizer,
    additional_num_tokens: int,
    max_token_length: int,
) -> list[UnigramToken]:
    base_tokens = {(t,) for t in pretokenizer.base_tokens}
    substring_freq = Counter({t: 0 for t in base_tokens})
    for base_token_seq, count in corpus:
        for i in range(len(base_token_seq)):  # SentencePiece uses suffix array, this is simpler but more mem intensive
            for j in range(i + 1, min(len(base_token_seq) + 1, i + max_token_length + 1)):
                if pretokenizer.token_allowed(base_token_seq[i:j]):  # char boundaries etc enforced here!
                    substring_freq[tuple(base_token_seq[i:j])] += count

    all_tokens = [(max(freq, 1) * len(token), token) for token, freq in substring_freq.items()]
    selected_tokens = heapq.nlargest(
        len(base_tokens) + additional_num_tokens, all_tokens, key=lambda item: (item[1] in base_tokens, item[0])
    )
    log_sum_scores = math.log(sum(score for score, _ in selected_tokens))
    tokens = [
        UnigramToken(
            base_tokens=token_array(base_token_seq),
            id=i,
            log_prob=math.log(score) - log_sum_scores,
            required=base_token_seq in base_tokens,
        )
        for i, (score, base_token_seq) in enumerate(selected_tokens)
    ]

    logger.info(
        f"🌱 Selected {len(pretokenizer.base_tokens):,} + {additional_num_tokens:,} = {len(tokens):,} initial tokens from {len(all_tokens):,} candidates"
    )
    logger.debug(f"   ├─ Max length: {max_token_length}")
    logger.debug(f"   └─ Source: {corpus.name}: {corpus.metadata}")
    return tokens


def run_e_step(
    logger: logging.Logger,
    corpus: PretokenizedCorpus,
    model: UnigramModel,
) -> tuple[dict[int, float], float, int]:
    """Performs the Expectation step of the EM algorithm for Unigram."""

    expected_count = defaultdict(float)
    objective = total_tokens = 0
    total_pretoken_freq = sum(freq for _, freq in corpus)

    if total_pretoken_freq == 0:
        return expected_count, objective, total_tokens

    for base_token_seq, freq in corpus:
        lattice = model.make_lattice(base_token_seq)
        z, token_prob = lattice.calc_marginal()
        assert not math.isnan(z), f"NaN likelihood for pretoken {base_token_seq} with freq={freq}."
        for token_id, prob in token_prob.items():
            expected_count[token_id] += prob * freq
        viterbi_path, _ = lattice.viterbi()
        num_tokens_in_sentence = len(viterbi_path)
        total_tokens += num_tokens_in_sentence * freq
        objective -= (z * freq) / total_pretoken_freq

    return expected_count, objective, total_tokens


def run_m_step(
    logger: logging.Logger,
    pretokenizer: Pretokenizer,
    model: UnigramModel,
    expected_count: dict[int, float],
    dp_smoothing: bool,
    k_expected_frequency_threshold: float,
) -> tuple[UnigramModel, int]:
    """Performs the Maximization step of the EM algorithm for Unigram.

    Args:
        expected_counts: Expected frequency for each token from E-step
        dp_smoothing: If True, use digamma-based sparsity (like SentencePiece).
                      If False, use standard maximum likelihood estimation.
    """
    # Filter infrequent pieces.
    filtered_tokens = [t for t in model.tokens if expected_count[t.id] >= k_expected_frequency_threshold or t.required]
    num_removed = len(model.tokens) - len(filtered_tokens)
    if num_removed > 0:
        filtered_ids = {t.id for t in filtered_tokens}
        removed_tokens = [(t, expected_count[t.id]) for t in model.tokens if t.id not in filtered_ids]
        logger.debug(
            f"   ├─ Removed {num_removed:,} low-frequency tokens below threshold {k_expected_frequency_threshold} - examples:"
        )
        log_examples(logger, pretokenizer, removed_tokens, "expected count")

        model = UnigramModel(pretokenizer, filtered_tokens)

    expected_count = {k: max(MIN_EXPECTED_COUNT, v) for k, v in expected_count.items()}
    total_freq = sum(expected_count[t.id] for t in model.tokens)
    if dp_smoothing:  # SentencePiece-style: digamma transform with implicit alpha=0 for sparsity bias
        log_total = digamma(total_freq)
        for t in model.tokens:
            t.log_prob = digamma(expected_count[t.id]) - log_total
    else:  # Standard maximum likelihood estimation
        for t in model.tokens:
            t.log_prob = math.log(expected_count[t.id] / total_freq)

    return model, num_removed


def prune_tokens(
    logger: logging.Logger,
    corpus: PretokenizedCorpus,
    pretokenizer: Pretokenizer,
    model: UnigramModel,
    desired_vocab_size: int,
    shrinking_factor: float,
    defensive: bool,
) -> tuple[UnigramModel, int, int, list[tuple[UnigramToken, float]]]:
    # Calculate target size based on vocab size and shrinking factor
    num_non_base_tokens = len(model.tokens) - len(pretokenizer.base_tokens)
    shrink_n = int(num_non_base_tokens * (1 - shrinking_factor))
    target_size = max(desired_vocab_size, num_non_base_tokens - shrink_n)

    # 1. Count occurences of all tokens in optimal tokenization of all pretokens
    token_count = {t.id: 0.0 for t in model.tokens}
    for base_token_seq, count in corpus:
        lattice = model.make_lattice(base_token_seq)
        viterbi_path, _ = lattice.viterbi()
        for token in viterbi_path:
            token_count[token.id] += count

    total_count = sum(token_count.values())
    log_total = math.log(total_count)
    # Second, consider which tokens we can remove and what the cost is
    candidates = []
    new_tokens = []
    unused_token_ids = set()
    for token in model.tokens:
        if token.required:  # never remove required tokens
            new_tokens.append(token)
            continue
        if token_count[token.id] == 0:  # never used in an optimal segmentation. includes split viterbi path.
            unused_token_ids.add(token.id)
            continue
        lattice = model.make_lattice(token.base_tokens)
        alt_path, _ = lattice.viterbi(allow_single_token=False)
        assert alt_path, f"Token {token.id} has no alternative segmentation"

        # Computes how the LM likelihood is reduced if the token is removed from the vocabulary.
        # Since the exact computation of loss is difficult, we compute the loss approximately by assuming that all
        # token[id] in the sentences are replaced with their second best segmentation (alt_path) when removed.

        # The logprob with the token[i] = log(count[i] / total_count)
        logprob_token = math.log(token_count[token.id]) - log_total
        # After removing the token[i], its frequency freq[i] is re-assigned to alternatives.
        # new_sum = current_sum - freq[i] + freq[i] * alternatives[i].size()
        #         = current_sum + freq[i] * (alternatives[i] - 1)
        logsum_alt = math.log(total_count + token_count[token.id] * (len(alt_path) - 1))
        # The frequencies of alternatives are increased by freq[i]
        logprob_alt = sum(math.log(token_count[alt.id] + token_count[token.id]) - logsum_alt for alt in alt_path)
        # loss: the diff of likelihood after removing the token[i]
        loss = (token_count[token.id] / total_count) * (logprob_token - logprob_alt)
        # (NEW FEATURE) if alternatives are already gone, optionally prevent removing this token
        defended = any(alt.id in unused_token_ids for alt in alt_path)
        candidates.append((token, loss, defended))

    # Finally, reduce vocabulary to target_size
    candidates.sort(key=lambda x: -x[1])
    defended_tokens = []
    for token, loss, defended in candidates:
        if len(new_tokens) < target_size:
            new_tokens.append(token)
        elif defensive and defended:
            defended_tokens.append((token, loss))
            new_tokens.append(token)

    new_token_ids = {t.id for t in new_tokens}
    pruned_tokens = [
        (token, loss)
        for token, loss, _ in candidates
        if token.id not in new_token_ids and token.id not in unused_token_ids
    ]

    logger.info(
        f"✂️  Pruning vocabulary from {len(model.tokens):,} to target {target_size:,} -> new vocab size {len(new_tokens):,}"
    )
    logger.debug(
        f"   ├─ Target size: {target_size:,} based on shrinking factor {shrinking_factor} * non base tokens {num_non_base_tokens:,} and desired vocab size {desired_vocab_size}"
    )
    if unused_token_ids:
        unused_tokens_info = [(model.tokens_by_id[tid], token_count[tid]) for tid in unused_token_ids]
        logger.debug(f"   ├─ Dropped {len(unused_tokens_info):,} tokens not in any optimal path")
        log_examples(logger, pretokenizer, unused_tokens_info, "count")
    logger.debug(f"   ├─ Kept {len(new_tokens)} required tokens")
    if defended_tokens:
        logger.debug(f"   ├─ Defended {len(defended_tokens):,} tokens from being removed along with their alternatives")
        log_examples(logger, pretokenizer, defended_tokens, "loss")

    logger.debug(f"   ├─ Pruned {len(pruned_tokens):,} tokens from {len(candidates):,} candidates")
    if pruned_tokens:
        log_examples(logger, pretokenizer, pruned_tokens, "loss")
    if candidates:
        logger.debug(f"   └─ Candidates loss range: {candidates[0][1]:.4g} to {candidates[-1][1]:.4g}")
    else:
        logger.info("   └─ No candidates for pruning!")

    return UnigramModel(model.pretokenizer, new_tokens), len(unused_token_ids), len(pruned_tokens), defended_tokens


def finalize_tokens(
    logger: logging.Logger,
    pretokenizer: Pretokenizer,
    model: UnigramModel,
    vocab_size: int,
) -> tuple[UnigramModel, int]:
    """Finalizes the vocabulary based on frequency in optimal tokenizations."""

    final_tokens = {}
    # Add required tokens
    for token in model.tokens:
        if token.required:
            final_tokens[token.id] = token

    # Keep highest scoring tokens
    for token in sorted(model.tokens, key=lambda x: -x.log_prob):
        if token.id in final_tokens:
            continue
        if len(final_tokens) >= vocab_size:
            break
        final_tokens[token.id] = token

    removed_tokens = [(t, t.log_prob) for t in model.tokens if t.id not in final_tokens]
    logger.info(f"✨ Finalizing vocabulary from {len(model.tokens):,} to target {vocab_size:,}")
    logger.info(f" ├─ Kept {len(final_tokens):,} tokens")
    logger.info(f" └─ Removed {len(removed_tokens):,} tokens")
    log_examples(logger, pretokenizer, removed_tokens, "logprob")

    new_model = UnigramModel(pretokenizer, list(final_tokens.values()))
    return new_model, len(model.tokens) - len(new_model.tokens)


# --- Main Training Function ---


def train_unigram(
    pretokenizer: Pretokenizer,
    corpus: PretokenizedCorpus,
    additional_vocab_size: int,
    num_workers: int = 1,  # TODO: unused/ignored for now
    verbose: bool = True,
    # unigram specific settings, some experimental
    max_token_len: int = 32,  # pretokenizer dependent?
    initial_vocab_factor: int = 10,
    pre_final_vocab_factor: float = 1.1,
    pruning_shrinking_factor: float = 0.75,
    m_step_dp_smoothing: bool = True,
    m_step_low_count_threshold: float = 0.5,
    defensive_prune: bool = False,
    max_iterations: int = 100,
    num_sub_iterations: int = 2,
) -> UnigramModel:
    """Trains a Unigram tokenizer model."""
    logger = create_logger("train_unigram", verbose)

    # initialize vocab and model
    vocab = make_initial_vocab(
        logger, corpus, pretokenizer, additional_vocab_size * initial_vocab_factor, max_token_len
    )
    total_pretokens = sum(freq for _, freq in corpus)
    prune_to_vocab_size = int(len(pretokenizer.base_tokens) + additional_vocab_size * pre_final_vocab_factor)
    final_vocab_size = len(pretokenizer.base_tokens) + additional_vocab_size
    totals_removed = defaultdict(list)
    defended_token_ids = set()

    model = UnigramModel(pretokenizer, vocab)

    # EM Training Loop
    for iter in range(max_iterations):
        # Sub-EM Iterations
        for sub_iter in range(num_sub_iterations):
            logger.info(f"🔄 EM Iteration {iter + 1}.{sub_iter + 1}. Model size {len(model.tokens):,}")
            expected_count, objective, total_tokens = run_e_step(logger=logger, corpus=corpus, model=model)
            model, m_step_removed = run_m_step(
                logger=logger,
                pretokenizer=pretokenizer,
                model=model,
                expected_count=expected_count,
                dp_smoothing=m_step_dp_smoothing,
                k_expected_frequency_threshold=m_step_low_count_threshold,
            )
            totals_removed["M Step Low Count"].append(m_step_removed)
            avg_tokens_per_pretoken = 1.0 * total_tokens / total_pretokens
            logger.debug(f"   ├─ Objective: {objective:.4f}")
            logger.debug(f"   ├─ Total tokens: {total_tokens:,d}")
            logger.debug(f"   └─ Avg tokens/pretoken: {avg_tokens_per_pretoken:.4f}")

        # Check Stopping Condition
        current_size = len(model.tokens)
        if current_size <= prune_to_vocab_size:
            logger.info(f"🎯 Target vocabulary size for EM iterations reached")
            logger.debug(f"   ├─ Current: {current_size:,}")
            logger.debug(f"   └─ Target:  {prune_to_vocab_size:,}")
            break

        # Pruning Step
        model, num_unused, num_pruned, defended_tokens = prune_tokens(
            logger=logger,
            corpus=corpus,
            pretokenizer=pretokenizer,
            model=model,
            desired_vocab_size=prune_to_vocab_size,
            shrinking_factor=pruning_shrinking_factor,
            defensive=defensive_prune,
        )
        totals_removed["Prune/Zero Count"].append(num_unused)
        totals_removed["Prune/Loss"].append(num_pruned)
        defended_token_ids.update(t.id for t, _ in defended_tokens)

    # Finalization
    model, finalize_removed = finalize_tokens(
        logger=logger, pretokenizer=pretokenizer, model=model, vocab_size=final_vocab_size
    )
    totals_removed["Finalize"].append(finalize_removed)

    expected_count, objective, total_tokens = run_e_step(logger, corpus=corpus, model=model)
    stats = {
        "objective": objective,
        "total_tokens": total_tokens,
        "tokens/pretoken": total_tokens / total_pretokens,
    }
    num_defended = len(defended_token_ids)
    defended_in_final = [(t, t.log_prob) for t in model.tokens if t.id in defended_token_ids]
    logger.info(f"🎉 Training completed successfully! Avg tokens/pretoken: {stats['tokens/pretoken']:.4f}")
    logger.debug(f"   ├─ Objective: {stats['objective']:.4f}")
    logger.debug("  📊 Token Removal Statistics:")
    for key, value in totals_removed.items():
        logger.debug(f"   ├─ {key:<20} {sum(value):6,d} tokens" + (f" in steps {value}" if len(value) > 1 else ""))
    if defensive_prune:
        logger.debug(f"   ├─ Defended {num_defended:,} tokens from being removed along with their alternatives.")
        if defended_in_final:
            logger.debug(f"   ├─ {len(defended_in_final):,} defended tokens made it to the final vocabulary.")
            log_examples(logger, pretokenizer, defended_in_final, "logprob")
        else:
            logger.debug("   ├─ No defended tokens made it to the final vocabulary.")
    logger.debug("  📊 Compression Statistics:")
    logger.debug(f"   ├─ Total tokens: {stats['total_tokens']:,d}")
    logger.debug(f"   └─ Avg tokens/pretoken: {stats['tokens/pretoken']:.4f}")

    model.metadata = stats
    return model

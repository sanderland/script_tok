from __future__ import annotations
from typing import Sequence

import heapq
import math
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from scipy.special import digamma

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import Pretokenizer
from script_bpe.tokenizers.base import BaseTrainer, TrainerConfig
from script_bpe.tokenizers.unigram.model import UnigramModel, UnigramToken
from script_bpe.tokenizers.unigram.init_algorithms import (
    compute_substring_frequencies_simple,
    compute_substring_frequencies_spm,
)
from script_bpe.tokenizers.unigram.init_corpus import (
    compute_substring_frequencies_corpus,
)
from script_bpe.utils import token_array
from pydantic import ConfigDict


class UnigramTrainerConfig(TrainerConfig):
	max_token_len: int = 32
	initial_vocab_factor: int = 10
	pre_final_vocab_factor: float = 1.1
	pruning_shrinking_factor: float = 0.75
	m_step_dp_smoothing: bool = True
	m_step_low_count_threshold: float = 0.5
	defensive_prune: bool = False
	# Use score-based pruning (like finalize) instead of Viterbi-based pruning during training.
	# This is faster but may be less effective at preserving a diverse, useful vocabulary.
	# The Viterbi approach considers alternative segmentations when deciding what to prune,
	# while score-based pruning only looks at token probabilities. In theory, score-based
	# pruning might work well if the M-step scores are already well-calibrated, but it risks
	# keeping redundant high-probability tokens while removing diverse lower-probability ones.
	final_style_prune: bool = False
	max_iterations: int = 100
	num_sub_iterations: int = 2
	init_vocab_algo: str = "spm_repair"  # one of {"simple", "spm", "spm_repair", "corpus_long", "corpus_intermediate", "corpus_repair"}
	forced_initial_vocab: Sequence[UnigramToken] | None = None
	model_config = ConfigDict(arbitrary_types_allowed=True)


class UnigramTrainer(BaseTrainer):
	MIN_EXPECTED_COUNT: float = 0.01

	def __init__(self, pretokenizer: Pretokenizer, corpus: PretokenizedCorpus, config: UnigramTrainerConfig):
		super().__init__(pretokenizer, corpus, config)

	def train(self) -> UnigramModel:
		cfg = self.config
		vocab, initial_vocab_size = self.make_initial_vocab()
		total_pretokens = sum(freq for _, freq in self.corpus)
		corpus_atomic_length = sum(len(atomic_token_seq) * freq for atomic_token_seq, freq in self.corpus)
		prune_to_vocab_size = int(
			len(self.pretokenizer.atomic_tokens) + cfg.additional_vocab_size * cfg.pre_final_vocab_factor
		)
		final_vocab_size = len(self.pretokenizer.atomic_tokens) + cfg.additional_vocab_size
		totals_removed = defaultdict(list)
		defended_token_ids = set()

		model = UnigramModel(self.pretokenizer, vocab)

		# Create thread pool once and reuse throughout training
		with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
			for iter in range(cfg.max_iterations):
				for sub_iter in range(cfg.num_sub_iterations):
					self.logger.info(f"🔄 EM Iteration {iter + 1}.{sub_iter + 1}. Model size {len(model.tokens):,}")
					expected_count, objective, total_tokens = self.run_e_step(model, corpus_atomic_length, executor)
					model, m_step_removed = self.run_m_step(model, expected_count)
					totals_removed["M Step Low Count"].append(m_step_removed)
					avg_tokens_per_pretoken = 1.0 * total_tokens / total_pretokens
					self.logger.debug(f"   ├─ Objective: {objective:.4f}")
					self.logger.debug(f"   ├─ Total tokens: {total_tokens:,d}")
					self.logger.debug(f"   └─ Avg tokens/pretoken: {avg_tokens_per_pretoken:.4f}")

				current_size = len(model.tokens)
				if current_size <= prune_to_vocab_size:
					self.logger.info("🎯 Target vocabulary size for EM iterations reached")
					self.logger.debug(f"   ├─ Current: {current_size:,}")
					self.logger.debug(f"   └─ Target:  {prune_to_vocab_size:,}")
					break

				model, num_unused, num_pruned, defended_tokens = self.prune_tokens(model, prune_to_vocab_size, executor)
				totals_removed["Prune/Zero Count"].append(num_unused)
				totals_removed["Prune/Loss"].append(num_pruned)
				defended_token_ids.update(t.id for t, _ in defended_tokens)

			model, finalize_removed = self.finalize_tokens(model, final_vocab_size)
			totals_removed["Finalize"].append(finalize_removed)

			expected_count, objective, total_tokens = self.run_e_step(model, corpus_atomic_length, executor)
		num_defended = len(defended_token_ids)
		defended_in_final = [(t, t.log_prob) for t in model.tokens.values() if t.id in defended_token_ids]
		stats = {
			"objective": objective,
			"total_tokens": total_tokens,
			"tokens/pretoken": total_tokens / total_pretokens,
			"num_iterations": iter + 1,
			"totals_removed": totals_removed,
			"num_defended": num_defended,
			"defended_in_final": defended_in_final,
			"initial_vocab_size": initial_vocab_size,
		}
		self.logger.info(f"🎉 Training completed successfully! Avg tokens/pretoken: {stats['tokens/pretoken']:.4f}")
		self.logger.debug(f"   ├─ Objective: {stats['objective']:.4f}")
		self.logger.debug("  📊 Token Removal Statistics:")
		for key, value in totals_removed.items():
			self.logger.debug(
				f"   ├─ {key:<20} {sum(value):6,d} tokens" + (f" in steps {value}" if len(value) > 1 else "")
			)
		if cfg.defensive_prune:
			self.logger.debug(
				f"   ├─ Defended {num_defended:,} tokens from being removed along with their alternatives."
			)
			if defended_in_final:
				self.logger.debug(f"   ├─ {len(defended_in_final):,} defended tokens made it to the final vocabulary.")
				self.log_examples(defended_in_final, "logprob")
			else:
				self.logger.debug("   ├─ No defended tokens made it to the final vocabulary.")
		self.logger.debug("  📊 Compression Statistics:")
		self.logger.debug(f"   ├─ Total tokens: {stats['total_tokens']:,d}")
		self.logger.debug(f"   └─ Avg tokens/pretoken: {stats['tokens/pretoken']:.4f}")

		model.metadata = stats
		return model

	# ---- Helper methods ----
	def log_examples(self, tokens_with_scores: list[tuple[UnigramToken, float]], score_label="score", n=5):
		tokens_with_scores.sort(key=lambda x: x[1])
		for i, (t, score) in enumerate(tokens_with_scores):
			if len(tokens_with_scores) > 2 * n:
				if i == n:
					self.logger.debug("   │  ├─ ...")
				if n < i < len(tokens_with_scores) - n:
					continue
			list_item = " ├─" if i < len(tokens_with_scores) - 1 else " └─"
			self.logger.debug(
				f"   │ {list_item} {repr(self.pretokenizer.decode(t.atomic_tokens,errors='backslashreplace')):25}  {score_label} = {score:10.3g}  atomic_tokens = {list(t.atomic_tokens)}"
			)


	@staticmethod
	def init_vocab_normalize_scores(all_tokens: list[tuple[float, tuple[int,...]]], n) -> list[UnigramToken]:
		selected_tokens = heapq.nlargest(n, all_tokens, key=lambda item: (len(item[1]) == 1, item[0]))
		log_sum_scores = math.log(sum(score for score, _ in selected_tokens))
		return [
			UnigramToken(
				atomic_tokens=token_array(atomic_token_seq),
				id=i,
				log_prob=math.log(score) - log_sum_scores,
				required=len(atomic_token_seq) == 1,
			)
			for i, (score, atomic_token_seq) in enumerate(selected_tokens)
		]

	def make_initial_vocab(self) -> tuple[list[UnigramToken], int]:
		if self.config.forced_initial_vocab:
			self.logger.info(f"🌱 Using {len(self.config.forced_initial_vocab)} forced initial tokens.")
			# The second element of the tuple is the number of candidates, which is not applicable here.
			return list(self.config.forced_initial_vocab), len(self.config.forced_initial_vocab)

		additional_num_tokens = self.config.additional_vocab_size * self.config.initial_vocab_factor
		max_token_length = self.config.max_token_len
		atomic_tokens = {(t,) for t in self.pretokenizer.atomic_tokens}

		algo = (self.config.init_vocab_algo or "spm").lower().strip()
		if algo == "simple":
			substring_freq = compute_substring_frequencies_simple(self.pretokenizer, self.corpus, max_token_length)
		elif algo == "spm":
			substring_freq = compute_substring_frequencies_spm(self.pretokenizer, self.corpus, max_token_length, repair=False)
		elif algo == "spm_repair":
			substring_freq = compute_substring_frequencies_spm(self.pretokenizer, self.corpus, max_token_length, repair=True)
		elif algo in ["corpus_long", "corpus_intermediate", "corpus_repair"]:
			# Convert to flat corpus of (tuple[int,...], int)
			flat_corpus = [(tuple(seq), freq) for seq, freq in self.corpus]
			substring_freq = compute_substring_frequencies_corpus(
				self.pretokenizer,
				flat_corpus,
				max_token_length,
				strategy=algo.removeprefix("corpus_"),
			)
			self.logger.info(f"🔍 Corpus based init vocab {algo!r}. Number of tokens: {len(substring_freq):,} tokens")
		else:
			raise ValueError(f"Unknown init_vocab_algo: {self.config.init_vocab_algo}. Use one of 'simple', 'spm', 'spm_repair', 'corpus_long', 'corpus_intermediate', 'corpus_repair'.")

		# Ensure atomic tokens are present (with at least frequency 0)
		for t in atomic_tokens:
			if t not in substring_freq:
				substring_freq[t] = 0
		all_tokens = [(max(freq, 1) * len(token), token) for token, freq in substring_freq.items()]
		tokens = self.init_vocab_normalize_scores(all_tokens, len(atomic_tokens) + additional_num_tokens)
		self.logger.info(
			f"🌱 Selected {len(self.pretokenizer.atomic_tokens):,} + {self.config.additional_vocab_size * self.config.initial_vocab_factor:,} = {len(tokens):,} initial tokens from {len(all_tokens):,} candidates"
		)
		self.logger.debug(f"   ├─ Max length: {max_token_length}")
		self.logger.debug(f"   └─ Source: {self.corpus.name}: {self.corpus.metadata}")
		return tokens, len(all_tokens)

	def run_e_step(self, model: UnigramModel, corpus_atomic_length: int, executor: ThreadPoolExecutor) -> tuple[dict[int, float], float, int]:
		"""Parallel E-step using threading."""
		num_workers = self.config.num_workers
		
		def worker_e_step(worker_id: int):
			local_expected_count = defaultdict(float)
			local_objective = 0.0
			local_total_tokens = 0
			
			for atomic_token_seq, freq in self.corpus.worker_iterate(worker_id, num_workers):
				lattice = model.make_lattice(atomic_token_seq)
				z, token_prob = lattice.calc_marginal()
				assert not math.isnan(z), f"NaN likelihood for pretoken {atomic_token_seq} with freq={freq}."
				for token_id, prob in token_prob.items():
					local_expected_count[token_id] += prob * freq
				viterbi_path, _ = lattice.viterbi()
				local_total_tokens += len(viterbi_path) * freq
				local_objective -= (z * freq)
			
			return local_expected_count, local_objective, local_total_tokens
		
		results = list(executor.map(worker_e_step, range(num_workers)))
		
		# Aggregate results
		expected_count = defaultdict(float)
		objective = 0.0
		total_tokens = 0
		
		for local_expected_count, local_objective, local_total_tokens in results:
			for token_id, count in local_expected_count.items():
				expected_count[token_id] += count
			objective += local_objective
			total_tokens += local_total_tokens
		
		objective /= corpus_atomic_length
		return expected_count, objective, total_tokens

	def run_m_step(self, model: UnigramModel, expected_count: dict[int, float]) -> tuple[UnigramModel, int]:
		dp_smoothing = self.config.m_step_dp_smoothing
		k_expected_frequency_threshold = self.config.m_step_low_count_threshold
		filtered_tokens = [
							t for t in model.tokens.values() if expected_count[t.id] >= k_expected_frequency_threshold or t.required
						]
		num_removed = len(model.tokens) - len(filtered_tokens)
		if num_removed > 0:
			filtered_ids = {t.id for t in filtered_tokens}
			removed_tokens = [(t, expected_count[t.id]) for t in model.tokens.values() if t.id not in filtered_ids]
			self.logger.debug(
				f"   ├─ Removed {num_removed:,} low-frequency tokens below threshold {k_expected_frequency_threshold} - examples:"
			)
			self.log_examples(removed_tokens, "expected count")
			model = UnigramModel(self.pretokenizer, filtered_tokens)
		expected_count = {k: max(self.MIN_EXPECTED_COUNT, v) for k, v in expected_count.items()}
		total_freq = sum(expected_count[t.id] for t in model.tokens.values())
		if dp_smoothing:
			log_total = digamma(total_freq)
			for t in model.tokens.values():
				t.log_prob = digamma(expected_count[t.id]) - log_total
		else:
			for t in model.tokens.values():
				t.log_prob = math.log(expected_count[t.id] / total_freq)
		return model, num_removed

	def score_based_prune(self, model: UnigramModel, target_size: int) -> tuple[UnigramModel, int]:
		"""
		Prune tokens based purely on their log probability scores.
		
		This is a simpler alternative to the Viterbi-based pruning that doesn't 
		consider alternative segmentations. Simply keeps the top-N tokens by log probability.
		
		Merit discussion:
		- PROS: Much faster (O(n log n) vs O(n * corpus_size)), simpler, deterministic
		- CONS: Ignores token redundancy and alternative segmentations
		
		The Viterbi-based approach is theoretically superior because it estimates the 
		*loss* from removing each token by comparing against alternative segmentations.
		This helps avoid removing tokens that are critical despite lower probability.
		
		However, score-based pruning might still work reasonably well if:
		1. The M-step has already calibrated probabilities well
		2. Speed is critical (e.g., very large vocabularies or corpora)
		3. The vocab is far from final size (early iterations)
		
		In practice, using score-based pruning during training iterations and Viterbi-based
		for final pruning might be a good compromise for speed vs quality.
		"""
		final_tokens = {}
		for token in model.tokens.values():
			if token.required:
				final_tokens[token.id] = token
		for token in sorted(model.tokens.values(), key=lambda x: -x.log_prob):
			if token.id in final_tokens:
				continue
			if len(final_tokens) >= target_size:
				break
			final_tokens[token.id] = token
		removed_tokens = [(t, t.log_prob) for t in model.tokens.values() if t.id not in final_tokens]
		self.logger.info(f"✂️  Score-based pruning from {len(model.tokens):,} to target {target_size:,}")
		self.logger.info(f"   ├─ Kept {len(final_tokens):,} tokens")
		self.logger.info(f"   └─ Removed {len(removed_tokens):,} tokens")
		self.log_examples(removed_tokens, "logprob")
		new_model = UnigramModel(model.pretokenizer, list(final_tokens.values()))
		return new_model, len(removed_tokens)

	def prune_tokens(self, model: UnigramModel, desired_vocab_size: int, executor: ThreadPoolExecutor):
		num_non_atomic_tokens = len(model.tokens) - len(self.pretokenizer.atomic_tokens)
		shrink_n = int(num_non_atomic_tokens * (1 - self.config.pruning_shrinking_factor))
		target_size = max(desired_vocab_size, num_non_atomic_tokens - shrink_n)
		
		# Use score-based pruning if configured
		if self.config.final_style_prune:
			new_model, num_removed = self.score_based_prune(model, target_size)
			# Return with zero defended tokens since score-based pruning doesn't use that concept
			return new_model, 0, num_removed, []
		
		# Otherwise use the Viterbi-based pruning approach
		num_workers = self.config.num_workers
		
		def worker_count_tokens(worker_id: int):
			local_token_count = defaultdict(float)
			for atomic_token_seq, count in self.corpus.worker_iterate(worker_id, num_workers):
				lattice = model.make_lattice(atomic_token_seq)
				viterbi_path, _ = lattice.viterbi()
				for token in viterbi_path:
					local_token_count[token.id] += count
			return local_token_count
		
		results = list(executor.map(worker_count_tokens, range(num_workers)))
		
		# Aggregate results
		token_count = {t.id: 0.0 for t in model.tokens.values()}
		for local_token_count in results:
			for token_id, count in local_token_count.items():
				token_count[token_id] += count
		total_count = sum(token_count.values())
		log_total = math.log(total_count) if total_count > 0 else float("-inf")
		candidates = []
		new_tokens = []
		unused_token_ids = set()
		for token in model.tokens.values():
			if token.required:
				new_tokens.append(token)
				continue
			if token_count[token.id] == 0:
				unused_token_ids.add(token.id)
				continue
			lattice = model.make_lattice(token.atomic_tokens)
			alt_path, _ = lattice.viterbi(allow_single_token=False)
			assert alt_path, f"Token {token.id} has no alternative segmentation"
			logprob_token = math.log(token_count[token.id]) - log_total
			logsum_alt = math.log(total_count + token_count[token.id] * (len(alt_path) - 1))
			logprob_alt = sum(math.log(token_count[alt.id] + token_count[token.id]) - logsum_alt for alt in alt_path)
			loss = (token_count[token.id] / total_count) * (logprob_token - logprob_alt)
			defended = any(alt.id in unused_token_ids for alt in alt_path)
			candidates.append((token, loss, defended))
		candidates.sort(key=lambda x: -x[1])
		defended_tokens = []
		for token, loss, defended in candidates:
			if len(new_tokens) < target_size:
				new_tokens.append(token)
			elif self.config.defensive_prune and defended:
				defended_tokens.append((token, loss))
				new_tokens.append(token)
		new_token_ids = {t.id for t in new_tokens}
		pruned_tokens = [
			(token, loss)
			for token, loss, _ in candidates
			if token.id not in new_token_ids and token.id not in unused_token_ids
		]
		self.logger.info(
			f"✂️  Pruning vocabulary from {len(model.tokens):,} to target {target_size:,} -> new vocab size {len(new_tokens):,}"
		)
		self.logger.debug(
			f"   ├─ Target size: {target_size:,} based on shrinking factor {self.config.pruning_shrinking_factor} * non base tokens {num_non_atomic_tokens:,} and desired vocab size {desired_vocab_size}"
		)
		if unused_token_ids:
			unused_tokens_info = [(model.tokens_by_id[tid], token_count[tid]) for tid in unused_token_ids]
			self.logger.debug(f"   ├─ Dropped {len(unused_tokens_info):,} tokens not in any optimal path")
			self.log_examples(unused_tokens_info, "count")
		self.logger.debug(f"   ├─ Kept {len(new_tokens)} required tokens")
		if defended_tokens:
			self.logger.debug(
				f"   ├─ Defended {len(defended_tokens):,} tokens from being removed along with their alternatives"
			)
			self.log_examples(defended_tokens, "loss")
		self.logger.debug(f"   ├─ Pruned {len(pruned_tokens):,} tokens from {len(candidates):,} candidates")
		if pruned_tokens:
			self.log_examples(pruned_tokens, "loss")
		if candidates:
			self.logger.debug(f"   └─ Candidates loss range: {candidates[0][1]:.4g} to {candidates[-1][1]:.4g}")
		else:
			self.logger.info("   └─ No candidates for pruning!")
		return UnigramModel(model.pretokenizer, new_tokens), len(unused_token_ids), len(pruned_tokens), defended_tokens

	def finalize_tokens(self, model: UnigramModel, vocab_size: int) -> tuple[UnigramModel, int]:
		"""Finalize the vocabulary using score-based pruning."""
		self.logger.info(f"✨ Finalizing vocabulary from {len(model.tokens):,} to target {vocab_size:,}")
		new_model, num_removed = self.score_based_prune(model, vocab_size)
		return new_model, num_removed

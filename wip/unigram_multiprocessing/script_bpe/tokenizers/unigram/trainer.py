from __future__ import annotations
from typing import Sequence

import heapq
import math
import multiprocessing
import time
from collections import Counter, defaultdict

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
from script_bpe.utils import token_array, mp_ctx
from pydantic import ConfigDict


def worker_process(
    worker_id: int,
    num_workers: int,
    pretokenizer: Pretokenizer,
    corpus: PretokenizedCorpus,
    cmd_queue: multiprocessing.Queue,
    results_queue: multiprocessing.Queue,
):
	"""Persistent worker process for Unigram training.
	
	Loads corpus partition once at startup and keeps it in memory.
	Processes commands from cmd_queue and returns results via results_queue.
	"""
	# Load corpus partition once and keep in memory
	t0_load = time.perf_counter()
	local_corpus_data = []
	total_freq = 0
	for atomic_token_seq, freq in corpus.worker_iterate(worker_id, num_workers):
		local_corpus_data.append((atomic_token_seq, freq))
		total_freq += freq
	del corpus  # Free corpus object after loading
	load_time = time.perf_counter() - t0_load
	results_queue.put(("STARTUP", worker_id, len(local_corpus_data), total_freq, load_time))
	
	while True:
		command = cmd_queue.get()
		
		if command[0] == "SHUTDOWN":
			break
			
		elif command[0] == "E_STEP":
			t0 = time.perf_counter()
			_, model = command
			
			# Model is already built in main process and pickled
			rebuild_time = 0.0
			
			local_expected_count = defaultdict(float)
			local_objective = 0.0
			local_total_tokens = 0
			num_lattices = 0
			
			t0_lattice = time.perf_counter()
			for atomic_token_seq, freq in local_corpus_data:
				num_lattices += 1
				lattice = model.make_lattice(atomic_token_seq)
				z, token_prob = lattice.calc_marginal()
				assert not math.isnan(z), f"NaN likelihood for pretoken {atomic_token_seq} with freq={freq}."
				for token_id, prob in token_prob.items():
					local_expected_count[token_id] += prob * freq
				viterbi_path, _ = lattice.viterbi()
				local_total_tokens += len(viterbi_path) * freq
				local_objective -= (z * freq)
			lattice_time = time.perf_counter() - t0_lattice
			total_time = time.perf_counter() - t0
			
			results_queue.put(("E_STEP", dict(local_expected_count), local_objective, local_total_tokens, 
			                  worker_id, rebuild_time, lattice_time, total_time, num_lattices))
			
		elif command[0] == "PRUNE":
			t0 = time.perf_counter()
			_, model = command
			
			# Model is already built in main process and pickled
			rebuild_time = 0.0
			
			local_token_count = defaultdict(float)
			
			t0_viterbi = time.perf_counter()
			for atomic_token_seq, count in local_corpus_data:
				lattice = model.make_lattice(atomic_token_seq)
				viterbi_path, _ = lattice.viterbi()
				for token in viterbi_path:
					local_token_count[token.id] += count
			viterbi_time = time.perf_counter() - t0_viterbi
			total_time = time.perf_counter() - t0
			
			results_queue.put(("PRUNE", dict(local_token_count), worker_id, rebuild_time, viterbi_time, total_time))


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
	init_vocab_algo: str = "corpus_repair"  # one of {"simple", "spm", "spm_repair", "corpus_long", "corpus_intermediate", "corpus_repair"}
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

		# Start persistent worker processes
		t0_startup = time.perf_counter()
		cmd_queues = [mp_ctx.Queue() for _ in range(cfg.num_workers)]
		results_queue = mp_ctx.Queue()
		workers = []
		
		for worker_id in range(cfg.num_workers):
			p = mp_ctx.Process(
				target=worker_process,
				args=(worker_id, cfg.num_workers, self.pretokenizer, self.corpus,
				      cmd_queues[worker_id], results_queue),
				daemon=True,
			)
			workers.append(p)
			p.start()
		
		# Wait for all workers to finish loading corpus
		worker_load_times = []
		for _ in range(cfg.num_workers):
			msg_type, worker_id, num_chunks, total_freq, load_time = results_queue.get()
			assert msg_type == "STARTUP"
			worker_load_times.append((worker_id, num_chunks, total_freq, load_time))
		
		total_startup = time.perf_counter() - t0_startup
		max_load_time = max(t for _, _, _, t in worker_load_times)
		avg_load_time = sum(t for _, _, _, t in worker_load_times) / len(worker_load_times)
		total_chunks = sum(c for _, c, _, _ in worker_load_times)
		total_pretokens = sum(f for _, _, f, _ in worker_load_times)
		self.logger.info(f"⏱️  Worker startup: {total_startup:.2f}s total, {max_load_time:.2f}s max load, {avg_load_time:.2f}s avg load")
		self.logger.info(f"   └─ Loaded {total_chunks:,} unique chunks, {total_pretokens:,} total pretokens across {cfg.num_workers} workers")
		for worker_id, num_chunks, total_freq, load_time in sorted(worker_load_times):
			self.logger.debug(f"      Worker {worker_id}: {num_chunks:,} chunks, {total_freq:,} pretokens in {load_time:.2f}s")
		
		try:
			for iter in range(cfg.max_iterations):
				for sub_iter in range(cfg.num_sub_iterations):
					self.logger.info(f"🔄 EM Iteration {iter + 1}.{sub_iter + 1}. Model size {len(model.tokens):,}")
					expected_count, objective, total_tokens = self.run_e_step(model, corpus_atomic_length, cmd_queues, results_queue)
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

				model, num_unused, num_pruned, defended_tokens = self.prune_tokens(model, prune_to_vocab_size, cmd_queues, results_queue)
				totals_removed["Prune/Zero Count"].append(num_unused)
				totals_removed["Prune/Loss"].append(num_pruned)
				defended_token_ids.update(t.id for t, _ in defended_tokens)

			model, finalize_removed = self.finalize_tokens(model, final_vocab_size)
			totals_removed["Finalize"].append(finalize_removed)

			expected_count, objective, total_tokens = self.run_e_step(model, corpus_atomic_length, cmd_queues, results_queue)
		finally:
			# Shutdown workers
			for q in cmd_queues:
				q.put(("SHUTDOWN",))
			for p in workers:
				p.join()
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

	def run_e_step(self, model: UnigramModel, corpus_atomic_length: int, cmd_queues, results_queue) -> tuple[dict[int, float], float, int]:
		"""Run E-step by sending commands to persistent workers."""
		t0_total = time.perf_counter()
		num_workers = self.config.num_workers
		
		# Send E_STEP command to all workers (send full model, not token list)
		t0_send = time.perf_counter()
		for q in cmd_queues:
			q.put(("E_STEP", model))
		send_time = time.perf_counter() - t0_send
		
		# Collect results from all workers
		t0_collect = time.perf_counter()
		expected_count = defaultdict(float)
		objective = 0.0
		total_tokens = 0
		worker_times = []
		
		for _ in range(num_workers):
			msg_type, local_expected_count, local_objective, local_total_tokens, worker_id, rebuild_time, lattice_time, worker_total, num_lattices = results_queue.get()
			assert msg_type == "E_STEP"
			for token_id, count in local_expected_count.items():
				expected_count[token_id] += count
			objective += local_objective
			total_tokens += local_total_tokens
			worker_times.append((worker_id, rebuild_time, lattice_time, worker_total, num_lattices))
		collect_time = time.perf_counter() - t0_collect
		total_time = time.perf_counter() - t0_total
		
		# Log detailed timing
		max_worker_time = max(t for _, _, _, t, _ in worker_times)
		avg_rebuild = sum(r for _, r, _, _, _ in worker_times) / len(worker_times)
		avg_lattice = sum(l for _, _, l, _, _ in worker_times) / len(worker_times)
		max_rebuild = max(r for _, r, _, _, _ in worker_times)
		max_lattice = max(l for _, _, l, _, _ in worker_times)
		total_lattices = sum(n for _, _, _, _, n in worker_times)
		
		self.logger.debug(f"   ⏱️  E-step timing: total={total_time:.2f}s, send={send_time:.3f}s, collect={collect_time:.3f}s, max_worker={max_worker_time:.2f}s")
		self.logger.debug(f"      ├─ Processed {total_lattices:,} unique lattices across {num_workers} workers")
		self.logger.debug(f"      └─ Worker avg: rebuild={avg_rebuild:.2f}s, lattice={avg_lattice:.2f}s | max: rebuild={max_rebuild:.2f}s, lattice={max_lattice:.2f}s")
		
		objective /= corpus_atomic_length
		return expected_count, objective, total_tokens

	def run_m_step(self, model: UnigramModel, expected_count: dict[int, float]) -> tuple[UnigramModel, int]:
		t0 = time.perf_counter()
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
		m_step_time = time.perf_counter() - t0
		self.logger.debug(f"   ⏱️  M-step timing: {m_step_time:.3f}s")
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

	def prune_tokens(self, model: UnigramModel, desired_vocab_size: int, cmd_queues, results_queue):
		t0_total = time.perf_counter()
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
		
		# Send PRUNE command to all workers (send full model)
		t0_send = time.perf_counter()
		for q in cmd_queues:
			q.put(("PRUNE", model))
		send_time = time.perf_counter() - t0_send
		
		# Collect results from all workers
		t0_collect = time.perf_counter()
		token_count = {t.id: 0.0 for t in model.tokens.values()}
		worker_times = []
		for _ in range(num_workers):
			msg_type, local_token_count, worker_id, rebuild_time, viterbi_time, worker_total = results_queue.get()
			assert msg_type == "PRUNE"
			for token_id, count in local_token_count.items():
				token_count[token_id] += count
			worker_times.append((worker_id, rebuild_time, viterbi_time, worker_total))
		collect_time = time.perf_counter() - t0_collect
		
		# Calculate loss for each candidate token
		t0_loss = time.perf_counter()
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
		loss_time = time.perf_counter() - t0_loss
		total_time = time.perf_counter() - t0_total
		
		new_token_ids = {t.id for t in new_tokens}
		pruned_tokens = [
			(token, loss)
			for token, loss, _ in candidates
			if token.id not in new_token_ids and token.id not in unused_token_ids
		]
		
		# Log detailed timing
		max_worker_time = max(t for _, _, _, t in worker_times)
		avg_rebuild = sum(r for _, r, _, _ in worker_times) / len(worker_times)
		avg_viterbi = sum(v for _, _, v, _ in worker_times) / len(worker_times)
		max_rebuild = max(r for _, r, _, _ in worker_times)
		max_viterbi = max(v for _, _, v, _ in worker_times)
		
		self.logger.info(
			f"✂️  Pruning vocabulary from {len(model.tokens):,} to target {target_size:,} -> new vocab size {len(new_tokens):,}"
		)
		self.logger.debug(f"   ⏱️  Prune timing: total={total_time:.2f}s, send={send_time:.3f}s, collect={collect_time:.3f}s, loss_calc={loss_time:.2f}s, max_worker={max_worker_time:.2f}s")
		self.logger.debug(f"      └─ Worker avg: rebuild={avg_rebuild:.2f}s, viterbi={avg_viterbi:.2f}s | max: rebuild={max_rebuild:.2f}s, viterbi={max_viterbi:.2f}s")
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

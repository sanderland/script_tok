"""ConvexTok trainer: flow-LP vocabulary selection (Tempus et al., 2026).

We build, per (aggregated) pretoken, the full segmentation DAG over its atomic
tokens and pose vocabulary selection as the linear program (paper Eq. 15-18):

    minimize    <w, f> + <w, g>                    # freq-weighted token count
    subject to  A f + B g = d                      # unit flow / valid path per word
                f - M t <= 0                        # use a token-instance only if its type is in V
                <1, t> <= K                         # vocabulary budget (non-atomic tokens)
                0 <= f, g, t <= 1

where ``f`` are *priced* (multi-atomic) token-instance edges, ``g`` are *free*
single-atomic edges (always available; the alphabet is locked in), ``t`` is the
per-type selection vector, ``M`` maps each priced edge to its token type, and
``d`` carries +1 at each word's start vertex and -1 at its end vertex.

The LP optimum is a lower bound on the corpus token count achievable by *any*
size-(|atomic|+K) vocabulary under optimal segmentation, so it certifies the
gap to optimality. We solve the relaxation with scipy's HiGHS backend (the
reference implementation uses GPU cuOpt), then round ``t`` to an integral
vocabulary (paper §: Det / Bias / Int, plus probabilistic). Inference is
PathPiece, identical to the reference's encoder -- see ``model.py``.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Literal

import numpy as np
from pydantic import ConfigDict
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, identity, vstack

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import Pretokenizer
from script_bpe.tokenizers.base import BaseTrainer, TrainerConfig
from script_bpe.tokenizers.convextok.model import ConvexTokModel
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.utils import token_array


class ConvexTokTrainerConfig(TrainerConfig):
    cmin: int = 100  # keep a candidate type only if its freq-weighted instance count exceeds this
    max_pretokens: int = 50_000  # top-N distinct pretokens used to build the flow LP (tractability)
    max_token_width: int = 16  # max candidate token width in atomic tokens (caps DAG edge count)
    rounding: Literal["det", "bias", "prob", "int"] = "det"
    lp_eps: float = 1e-6
    presolve: bool = True
    lp_time_limit: float | None = None
    # Solver backend. "highs"/"highs-ipm"/"highs-ds" use scipy's HiGHS (exact, but
    # IPM/simplex scale poorly past a few million vars). "pdlp" uses OR-Tools'
    # first-order PDLP (CPU), which scales to the full flow LP where HiGHS stalls
    # -- the same algorithm class the reference repo runs on GPU via cuOpt.
    lp_method: Literal["highs", "highs-ipm", "highs-ds", "pdlp"] = "highs"
    pdlp_tol: float | None = 1e-6  # PDLP relative/absolute optimality tolerance (None = solver default)
    model_config = ConfigDict(extra="forbid")


class ConvexTokTrainer(BaseTrainer):
    def __init__(self, pretokenizer: Pretokenizer, corpus: PretokenizedCorpus, config: ConvexTokTrainerConfig):
        super().__init__(pretokenizer, corpus, config)

    # ------------------------------------------------------------------
    def train(self) -> ConvexTokModel:
        cfg = self.config
        atomic_ids = sorted(self.pretokenizer.atomic_tokens.keys())
        n_atomic = len(atomic_ids)
        budget = cfg.additional_vocab_size  # K: number of non-atomic tokens the LP may select

        pretokens = self._select_pretokens(cfg.max_pretokens)
        self.logger.info(f"Using {len(pretokens):,} distinct pretokens to build the flow LP")

        cand_index = self._count_candidates(pretokens, cfg.cmin, cfg.max_token_width)
        n_cand = len(cand_index)
        self.logger.info(
            f"Candidate token types |T|={n_cand:,} (cmin={cfg.cmin}, L={cfg.max_token_width}), budget K={budget:,}"
        )
        if budget > n_cand:
            raise ValueError(
                f"Budget K={budget:,} exceeds candidate types |T|={n_cand:,}. "
                f"Lower additional_vocab_size, lower cmin, or raise max_token_width/max_pretokens."
            )

        index_to_tok: list[tuple] = [None] * n_cand  # type: ignore[list-item]
        for tok, idx in cand_index.items():
            index_to_tok[idx] = tok

        t_star, obj, status = self._build_and_solve(pretokens, cand_index, n_cand, budget)

        chosen = self._round(t_star, index_to_tok, budget, cfg.rounding, cfg.lp_eps)
        self.logger.info(f"Rounding={cfg.rounding}: selected {len(chosen):,} non-atomic tokens")

        tokens: list[UnigramToken] = [
            UnigramToken(id=aid, atomic_tokens=token_array([aid]), log_prob=0.0, required=True)
            for aid in atomic_ids
        ]
        next_id = (max(atomic_ids) if atomic_ids else -1) + 1
        for tok in chosen:
            tokens.append(UnigramToken(id=next_id, atomic_tokens=token_array(tok), log_prob=0.0, required=False))
            next_id += 1

        model = ConvexTokModel(self.pretokenizer, tokens)
        ctc = self._corpus_token_count(model, pretokens)
        # LP objective is a lower bound on the freq-weighted token count; the rounded
        # vocabulary's actual count (under PathPiece) is an upper bound. Their ratio
        # certifies how far this tokenizer is from optimal (>= 1.0; 1.0 == optimal).
        gap = (ctc / obj) if obj > 0 else float("nan")
        model.metadata = {
            "tokenizer_variant": "convextok",
            "cmin": cfg.cmin,
            "max_pretokens": cfg.max_pretokens,
            "max_token_width": cfg.max_token_width,
            "rounding": cfg.rounding,
            "lp_status": status,
            "lp_objective_lower_bound": obj,
            "rounded_corpus_token_count": ctc,
            "optimality_ratio": gap,
            "n_candidates": n_cand,
            "n_pretokens": len(pretokens),
            "final_vocab_size": len(model.tokens),
            "config": cfg.model_dump(),
        }
        self.logger.info(
            f"ConvexTok done. |V|={len(model.tokens):,} CTC={ctc:,} "
            f"(LP lower bound {obj:,.0f}, ratio {gap:.4f}, {status})"
        )
        return model

    # ------------------------------------------------------------------
    def _select_pretokens(self, max_pretokens: int) -> list[tuple[tuple, int]]:
        pretoks: list[tuple[tuple, int]] = [
            (tuple(memoryview(chunk).tolist()), freq) for chunk, freq in self.corpus if len(chunk) > 0
        ]
        pretoks.sort(key=lambda kv: -kv[1])
        return pretoks[:max_pretokens]

    def _count_candidates(self, pretokens: list[tuple[tuple, int]], cmin: int, L: int) -> dict[tuple, int]:
        """Freq-weighted instance count of every width-2..L substring; keep those over ``cmin``.

        Returns a stable {token_tuple -> column index} map (frequent types first).
        """
        counter: Counter[tuple] = Counter()
        for seq, freq in pretokens:
            n = len(seq)
            upper = min(L, n)
            for w in range(2, upper + 1):
                for i in range(n - w + 1):
                    counter[seq[i : i + w]] += freq
        cand = [tok for tok, c in counter.items() if c > cmin]
        cand.sort(key=lambda tok: (-counter[tok], tok))
        return {tok: i for i, tok in enumerate(cand)}

    # ------------------------------------------------------------------
    def _build_and_solve(self, pretokens, cand_index, n_cand, budget):
        """Assemble the flow LP standard form and solve with HiGHS.

        Variable order: [ f (priced edges) | g (free edges) | t (token types) ].
        """
        A_rows: list[int] = []  # priced-edge incidence (vertices x num_f)
        A_cols: list[int] = []
        A_data: list[float] = []
        B_rows: list[int] = []  # free-edge incidence (vertices x num_g)
        B_cols: list[int] = []
        B_data: list[float] = []
        M_rows: list[int] = []  # priced edge -> token type (num_f x n_cand)
        M_cols: list[int] = []
        M_data: list[float] = []
        b_parts: list[np.ndarray] = []
        wf_parts: list[np.ndarray] = []
        wg_parts: list[np.ndarray] = []

        v_off = 0  # vertex row offset
        f_off = 0  # priced-edge column offset
        g_off = 0  # free-edge column offset
        L = self.config.max_token_width

        for seq, freq in pretokens:
            n = len(seq)
            nv = n + 1
            local_f = 0
            upper = min(L, n)
            for w in range(2, upper + 1):
                for i in range(n - w + 1):
                    ti = cand_index.get(seq[i : i + w])
                    if ti is None:
                        continue
                    col = f_off + local_f
                    A_rows.append(i + v_off)
                    A_cols.append(col)
                    A_data.append(1.0)
                    A_rows.append(i + w + v_off)
                    A_cols.append(col)
                    A_data.append(-1.0)
                    M_rows.append(col)
                    M_cols.append(ti)
                    M_data.append(1.0)
                    local_f += 1
            # free edges: single atomic token i -> i+1 (always available)
            for i in range(n):
                col = g_off + i
                B_rows.append(i + v_off)
                B_cols.append(col)
                B_data.append(1.0)
                B_rows.append(i + 1 + v_off)
                B_cols.append(col)
                B_data.append(-1.0)

            b = np.zeros(nv, dtype=float)
            b[0] = 1.0
            b[nv - 1] = -1.0
            b_parts.append(b)
            wf_parts.append(np.full(local_f, float(freq), dtype=float))
            wg_parts.append(np.full(n, float(freq), dtype=float))

            v_off += nv
            f_off += local_f
            g_off += n

        num_f, num_g, num_v = f_off, g_off, v_off
        N = num_f + num_g + n_cand
        self.logger.info(
            f"LP: {N:,} vars ({num_f:,} priced + {num_g:,} free + {n_cand:,} token) "
            f"{num_v:,} flow + {num_f:,} link + 1 budget constraints"
        )

        A = coo_matrix((A_data, (A_rows, A_cols)), shape=(num_v, num_f)).tocsr()
        B = coo_matrix((B_data, (B_rows, B_cols)), shape=(num_v, num_g)).tocsr()
        M = coo_matrix((M_data, (M_rows, M_cols)), shape=(num_f, n_cand)).tocsr()
        b_eq = np.concatenate(b_parts) if b_parts else np.zeros(0)

        # Equality: A f + B g = d   (token columns absent => zeros).
        A_eq = hstack([A, B, csr_matrix((num_v, n_cand))], format="csr")

        # Inequality 1 (linking): f - M t <= 0.
        A_ub_link = hstack([identity(num_f, format="csr"), csr_matrix((num_f, num_g)), -M], format="csr")
        # Inequality 2 (budget): sum(t) <= K.
        bud_cols = np.arange(num_f + num_g, N)
        A_ub_bud = coo_matrix(
            (np.ones(n_cand), (np.zeros(n_cand, dtype=int), bud_cols)), shape=(1, N)
        ).tocsr()
        A_ub = vstack([A_ub_link, A_ub_bud], format="csr")
        b_ub = np.concatenate([np.zeros(num_f), [float(budget)]])

        c = np.concatenate(
            [
                np.concatenate(wf_parts) if wf_parts else np.zeros(0),
                np.concatenate(wg_parts) if wg_parts else np.zeros(0),
                np.zeros(n_cand),
            ]
        )
        sf = {
            "c": c, "N": N, "num_f": num_f, "num_g": num_g, "n_cand": n_cand,
            "A_eq": A_eq, "b_eq": b_eq, "A_ub": A_ub, "b_ub": b_ub,
        }
        if self.config.lp_method == "pdlp":
            return self._solve_pdlp(sf)
        return self._solve_highs(sf)

    def _solve_highs(self, sf):
        t0 = time.perf_counter()
        res = linprog(
            sf["c"], A_ub=sf["A_ub"], b_ub=sf["b_ub"], A_eq=sf["A_eq"], b_eq=sf["b_eq"],
            bounds=[(0.0, 1.0)] * sf["N"], method=self.config.lp_method,
            options={"presolve": self.config.presolve, "time_limit": self.config.lp_time_limit},
        )
        if not res.success:
            raise RuntimeError(f"LP failed ({res.status}): {res.message}")
        self.logger.info(
            f"LP solved in {time.perf_counter() - t0:.1f}s ({self.config.lp_method}): obj={res.fun:,.0f}"
        )
        return res.x[sf["num_f"] + sf["num_g"] :], float(res.fun), f"highs:{res.status}"

    def _solve_pdlp(self, sf):
        """Solve the LP with OR-Tools PDLP (first-order, CPU-parallel).

        Builds the model in one vectorized C++ call (``fill_model_from_sparse_data``)
        from a single stacked constraint matrix: equality (flow) rows use lb=ub=d;
        inequality (linking + budget) rows use lb=-inf, ub=b_ub.
        """
        import pandas as pd
        from ortools.linear_solver.python import model_builder as mb

        N, num_f, num_g = sf["N"], sf["num_f"], sf["num_g"]
        n_ub = sf["A_ub"].shape[0]
        con_mat = vstack([sf["A_eq"], sf["A_ub"]], format="csr").astype(np.float64)
        con_lb = np.concatenate([sf["b_eq"], np.full(n_ub, -np.inf)])
        con_ub = np.concatenate([sf["b_eq"], sf["b_ub"]])

        model = mb.Model()
        model.helper.fill_model_from_sparse_data(
            np.zeros(N), np.ones(N), sf["c"].astype(np.float64), con_lb, con_ub, con_mat
        )
        model.helper.set_maximize(False)

        solver = mb.Solver("PDLP")
        if self.config.lp_time_limit:
            solver.set_time_limit_in_seconds(float(self.config.lp_time_limit))
        params = [f"num_threads: {max(1, self.config.num_workers)}"]
        if self.config.pdlp_tol is not None:
            params.append(
                f"termination_criteria {{ simple_optimality_criteria {{ "
                f"eps_optimal_relative: {self.config.pdlp_tol} "
                f"eps_optimal_absolute: {self.config.pdlp_tol} }} }}"
            )
        solver.set_solver_specific_parameters(" ".join(params))

        t0 = time.perf_counter()
        status = solver.solve(model)
        dt = time.perf_counter() - t0
        if status not in (mb.SolveStatus.OPTIMAL, mb.SolveStatus.FEASIBLE):
            raise RuntimeError(f"PDLP solve failed: status={status} ({solver.status_string})")
        obj = float(solver.objective_value)
        self.logger.info(f"LP solved in {dt:.1f}s (pdlp): status={status!s} obj={obj:,.0f}")
        t_vars = [model.var_from_index(i) for i in range(num_f + num_g, N)]
        t_star = solver.values(pd.Series(t_vars)).to_numpy()
        return t_star, obj, f"pdlp:{status!s}"

    # ------------------------------------------------------------------
    def _round(self, t_star, index_to_tok, budget, method, eps) -> list[tuple]:
        n = len(t_star)
        if method == "det":
            order = np.argsort(-t_star, kind="stable")
            return [index_to_tok[i] for i in order[:budget]]
        if method == "bias":
            # rank by LP mass per unit token length -> favours shorter, "surer" tokens
            score = np.array([t_star[i] / len(index_to_tok[i]) for i in range(n)], dtype=float)
            order = np.argsort(-score, kind="stable")
            return [index_to_tok[i] for i in order[:budget]]
        if method == "int":
            # keep only (near-)integral types; may yield fewer than `budget`
            idx = [i for i in range(n) if t_star[i] > 1.0 - eps]
            idx.sort(key=lambda i: -t_star[i])
            return [index_to_tok[i] for i in idx[:budget]]
        if method == "prob":
            return self._probabilistic_round(t_star, index_to_tok, budget, eps)
        raise ValueError(f"Unknown rounding method: {method!r}")

    @staticmethod
    def _probabilistic_round(t_star, index_to_tok, budget, eps) -> list[tuple]:
        """Keep (near-)integral types, then weighted-sample the rest by LP mass
        without replacement via the Gumbel-top-k trick (cf. reference impl)."""
        n = len(t_star)
        always = [i for i in range(n) if t_star[i] > 1.0 - eps]
        chosen = [index_to_tok[i] for i in always]
        remaining = budget - len(always)
        if remaining <= 0:
            return chosen[:budget]
        cand = [i for i in range(n) if i not in set(always) and t_star[i] > 0.0]
        if not cand:
            return chosen
        w = np.array([t_star[i] for i in cand], dtype=float)
        gumbel = -np.log(-np.log(np.random.random(len(cand))))
        keys = np.log(w) + gumbel
        k = min(remaining, len(cand))
        top = np.argpartition(-keys, k - 1)[:k]
        chosen.extend(index_to_tok[cand[j]] for j in top)
        return chosen

    # ------------------------------------------------------------------
    def _corpus_token_count(self, model: ConvexTokModel, pretokens: list[tuple[tuple, int]]) -> int:
        total = 0
        for seq, freq in pretokens:
            total += len(model.encode_chunk(token_array(seq))) * freq
        return total


__all__ = ["ConvexTokTrainer", "ConvexTokTrainerConfig"]

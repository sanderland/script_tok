"""ConvexTok tokenizer model (Tempus, Whittington, Schmidt, Komm, Pimentel, 2026;
"Tokenisation via Convex Relaxations", arXiv:2605.22821).

ConvexTok *selects* a vocabulary by solving a flow-based LP relaxation of the
vocabulary-selection IP (see ``trainer.py``). At inference time it segments
each pretoken with the compression-optimal minimum-token shortest-path DP --
which is exactly PathPiece (Schmidt et al., 2024), the encoder the paper uses.

We therefore reuse ``PathPieceModel``'s segmentation/serialisation verbatim and
only specialise the version tag and report title, so that saved ConvexTok
tokenizers detect and round-trip back to this class. The selection algorithm
(the paper's actual contribution) lives entirely in ``ConvexTokTrainer``.
"""

from script_bpe.tokenizers.pathpiece.model import PathPieceModel


class ConvexTokModel(PathPieceModel):
    VERSION = "seconvextok-v1"
    REPORT_TITLE = "ConvexTok Tokenizer Report"


__all__ = ["ConvexTokModel"]

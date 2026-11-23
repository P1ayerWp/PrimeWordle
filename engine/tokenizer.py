"""Compact 80-token symbolic vocabulary and trajectory serializer for PrimeWordle.

Kept alongside the core engine from Day 1 so human games, solver trajectories,
SFT datasets, and RL rollouts share an identical token representation.

Default vocabulary (max_symbols=32 -> vocab_size=80, aligned to multiple of 16):
  - 16 structural / rule / feedback tokens:
      <PAD>, <BOS>, <EOS>, <TURN>, <SLOT>, <FB>,
      <CFG>, <UNDERFILL_ON>, <UNDERFILL_OFF>, <DUP_ON>, <DUP_OFF>, <UNK_CFG>,
      EXACT, MISPLACED, PARTIAL, WRONG
  - 32 abstract prime symbol tokens: S_0 .. S_31
  - 32 config integer tokens:        N_1 .. N_32

Fixed-position 7-token config header:
  [0] <BOS>
  [1] <CFG>
  [2] N_P
  [3] N_L
  [4] N_K                (or <UNK_CFG> if reveal_k=False)
  [5] <UNDERFILL_ON/OFF> (or <UNK_CFG> if reveal_underfill=False)
  [6] <DUP_ON/OFF>       (or <UNK_CFG> if reveal_dup=False)
"""

from __future__ import annotations

from dataclasses import dataclass

from .game import Feedback, GameConfig, GameSession, TurnRecord


@dataclass
class PrimeWordleTokenizer:
    """Tiny fixed vocabulary: 16 specials/feedbacks + P symbols + P config numbers."""

    max_symbols: int = 32

    def __post_init__(self) -> None:
        specials = [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<TURN>",
            "<SLOT>",
            "<FB>",
            # Fixed-format config header tokens
            "<CFG>",
            "<UNDERFILL_ON>",
            "<UNDERFILL_OFF>",
            "<DUP_ON>",
            "<DUP_OFF>",
            "<UNK_CFG>",
            # 4 feedback states
            Feedback.EXACT.value,
            Feedback.MISPLACED.value,
            Feedback.PARTIAL.value,
            Feedback.WRONG.value,
        ]
        symbols = [f"S_{i}" for i in range(self.max_symbols)]
        numbers = [f"N_{i}" for i in range(1, self.max_symbols + 1)]

        self.itos: list[str] = specials + symbols + numbers
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<EOS>"]

    def encode_config(
        self,
        config: GameConfig,
        reveal_k: bool = True,
        reveal_underfill: bool = True,
        reveal_dup: bool = True,
    ) -> list[int]:
        """Encode fixed-position 7-token game header; hidden fields use <UNK_CFG>."""
        k_tok = f"N_{config.max_multiplier}" if reveal_k else "<UNK_CFG>"
        underfill_tok = (
            ("<UNDERFILL_ON>" if config.allow_underfill else "<UNDERFILL_OFF>")
            if reveal_underfill
            else "<UNK_CFG>"
        )
        dup_tok = (
            ("<DUP_ON>" if config.allow_duplicate_slots else "<DUP_OFF>")
            if reveal_dup
            else "<UNK_CFG>"
        )
        tokens = [
            "<BOS>",
            "<CFG>",
            f"N_{config.num_primes}",
            f"N_{config.code_length}",
            k_tok,
            underfill_tok,
            dup_tok,
        ]
        return [self.stoi[t] for t in tokens]

    def encode_turn(self, turn: TurnRecord) -> list[int]:
        """Encode one turn: <TURN> (<SLOT> S_a S_b ...)*L <FB> FB_0 ... FB_{L-1}."""
        tokens = ["<TURN>"]
        for slot in turn.guess_slots:
            tokens.append("<SLOT>")
            for sym_idx in slot:
                tokens.append(f"S_{sym_idx}")
        tokens.append("<FB>")
        for fb in turn.feedback:
            tokens.append(fb.value)
        return [self.stoi[t] for t in tokens]

    def encode_session(
        self,
        session: GameSession,
        reveal_k: bool = True,
        reveal_underfill: bool = True,
        reveal_dup: bool = True,
    ) -> list[int]:
        ids, _ = self.encode_session_with_loss_mask(
            session,
            reveal_k=reveal_k,
            reveal_underfill=reveal_underfill,
            reveal_dup=reveal_dup,
        )
        return ids

    def encode_session_with_loss_mask(
        self,
        session: GameSession,
        reveal_k: bool = True,
        reveal_underfill: bool = True,
        reveal_dup: bool = True,
    ) -> tuple[list[int], list[int]]:
        """Encode full session plus binary loss mask (1 = model action token, 0 = prompt/env).

        Model predicts:
          - Each turn's guess: `<TURN> (<SLOT> S_a ...)*L <FB>` (including `<FB>` as submit token)
          - `<EOS>` immediately after a winning all-EXACT feedback row.
        Fixed 7-token `<CFG>` header and environment feedbacks are masked out (0).
        """
        header = self.encode_config(
            session.config,
            reveal_k=reveal_k,
            reveal_underfill=reveal_underfill,
            reveal_dup=reveal_dup,
        )
        ids: list[int] = list(header)
        mask: list[int] = [0] * len(header)

        for turn in session.history:
            guess_tokens = ["<TURN>"]
            for slot in turn.guess_slots:
                guess_tokens.append("<SLOT>")
                for sym_idx in slot:
                    guess_tokens.append(f"S_{sym_idx}")
            guess_tokens.append("<FB>")
            guess_ids = [self.stoi[t] for t in guess_tokens]
            ids.extend(guess_ids)
            mask.extend([1] * len(guess_ids))

            fb_ids = [self.stoi[fb.value] for fb in turn.feedback]
            ids.extend(fb_ids)
            mask.extend([0] * len(fb_ids))

        if session.status == "won":
            ids.append(self.eos_id)
            mask.append(1)

        return ids, mask

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]

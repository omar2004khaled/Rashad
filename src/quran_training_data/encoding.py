"""Label ID mapping copied in behavior from upstream Muaalem vocab/tokenizer."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, get_args

from quran_transcript import SifaOutput, alphabet


PAD_ID = 0
CTC_BLANK_ID = PAD_ID
IGNORE_INDEX = -100


class UnknownLabelError(ValueError):
    """A source annotation contains a label absent from the model output head."""


class LabelCodec:
    """Strict char-level phoneme and per-sifat-level IDs.

    ID zero is reserved for [PAD] and is also the model's CTC blank. Unknown
    labels are rejected: the saved checkpoint's CTC heads cannot emit its
    tokenizer-only ``<unk>`` special token.
    """

    def __init__(self) -> None:
        symbols = list(asdict(alphabet.phonetics).values())
        if len(symbols) != len(set(symbols)):
            raise RuntimeError("quran-transcript phoneme alphabet contains duplicate symbols")
        self.phoneme_to_id = {symbol: i + 1 for i, symbol in enumerate(symbols)}
        self.id_to_phoneme = {value: key for key, value in self.phoneme_to_id.items()}
        self.sifat_to_id: dict[str, dict[str, int]] = {}
        self.id_to_sifat: dict[str, dict[int, str]] = {}
        for field_name, field in SifaOutput.model_fields.items():
            if field_name == "phonemes":
                continue
            categories = get_args(field.annotation)
            if not categories:
                raise RuntimeError(f"Unsupported reference sifat type for {field_name!r}")
            self.sifat_to_id[field_name] = {label: index + 1 for index, label in enumerate(categories)}
            self.id_to_sifat[field_name] = {index + 1: label for index, label in enumerate(categories)}

    @property
    def phoneme_vocab_size(self) -> int:
        return len(self.phoneme_to_id) + 1  # includes PAD/CTC blank

    @property
    def level_vocab_sizes(self) -> dict[str, int]:
        return {"phonemes": self.phoneme_vocab_size,
                **{name: len(labels) + 1 for name, labels in self.sifat_to_id.items()}}

    def encode_phonemes(self, text: str, *, remove_spaces: bool = False) -> list[int]:
        if not isinstance(text, str) or not text:
            raise UnknownLabelError("phoneme text must be a non-empty string")
        if remove_spaces:
            # quran-transcript.quran_phonetizer(remove_spaces=True) removes its
            # inter-word space marker before the original CTC tokenizer runs.
            text = text.replace(" ", "")
        if not text:
            raise UnknownLabelError("phoneme text contains no labels after space removal")
        output: list[int] = []
        for index, symbol in enumerate(text):
            try:
                output.append(self.phoneme_to_id[symbol])
            except KeyError as exc:
                raise UnknownLabelError(
                    f"Unsupported phoneme symbol {symbol!r} at Unicode index {index}; "
                    "no output class exists for it"
                ) from exc
        return output

    def decode_phonemes(self, ids: list[int] | tuple[int, ...]) -> str:
        chars: list[str] = []
        for index, token_id in enumerate(ids):
            if token_id == PAD_ID or token_id == IGNORE_INDEX:
                raise UnknownLabelError(f"Padding/blank id {token_id} is not a target label (index {index})")
            try:
                chars.append(self.id_to_phoneme[int(token_id)])
            except (KeyError, ValueError, TypeError) as exc:
                raise UnknownLabelError(f"Unknown phoneme id {token_id!r} at index {index}") from exc
        return "".join(chars)

    def encode_sifat(self, entries: list[Mapping[str, Any]]) -> dict[str, list[int]]:
        if not isinstance(entries, list) or not entries:
            raise UnknownLabelError("sifat must be a non-empty list of reference groups")
        encoded: dict[str, list[int]] = {name: [] for name in self.sifat_to_id}
        for group_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise UnknownLabelError(f"sifat group {group_index} is not a mapping")
            for field, labels in self.sifat_to_id.items():
                value = entry.get(field)
                if value not in labels:
                    raise UnknownLabelError(
                        f"Unsupported category {value!r} for {field!r} at group {group_index}"
                    )
                encoded[field].append(labels[value])
        return encoded

    def decode_sifat(self, field: str, ids: list[int] | tuple[int, ...]) -> list[str]:
        if field not in self.id_to_sifat:
            raise UnknownLabelError(f"Unknown sifat level {field!r}")
        result: list[str] = []
        for index, token_id in enumerate(ids):
            if token_id == PAD_ID or token_id == IGNORE_INDEX:
                raise UnknownLabelError(f"Padding/blank id {token_id} is not a target label (index {index})")
            try:
                result.append(self.id_to_sifat[field][int(token_id)])
            except (KeyError, ValueError, TypeError) as exc:
                raise UnknownLabelError(f"Unknown id {token_id!r} for sifat level {field!r}") from exc
        return result

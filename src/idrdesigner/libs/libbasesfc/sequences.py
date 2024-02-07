"""
Base sequence class and point mutations.

Caches data relevant for speeding up calculations of features based on a
previous sequence one point mutation away.

Author: Tian Hao Huang (@alex-dsci)
Date: Feb 3rd, 2024
"""

import json
from pathlib import Path
from typing import Optional, Collection, Any, Union
from copy import copy
import pprint

from idrdesigner.core.consts import AMINOACIDS, CHARGED_RES
from idrdesigner.core.exceptions import IDRDesignerException, UnreachableCode
from idrdesigner.libs import assert_valid_sequence, AlphabetValueException
from idrdesigner.libs.libbasesfc import default_config_path


class BaseExtendedSequence:  # pylint: disable=too-few-public-methods
    """
    Represents a sequence and stores data relevant for speeding up calculations
    of features based on previous sequences one point mutation away.
    """

    seq: str
    _feat_keys_from_config: dict[Path, Collection[str]] = {}
    """
    A static property which prevents
    redundant reloading of feature keys from a file.
    """
    feat_cache: dict[str, Optional[float]]
    """
    A cache of all features supported by the calculator,
    plus private builtin counts of all amino acids.
    """
    procharged_res: list[int]
    """Cached indices of prolines and charged residues."""
    charged_res: list[int]
    """Cached indices of charged residues."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        seq: "Union[str, BaseSeqRepr]",
        *,
        feat_config_path: Path = default_config_path,
        override_feat_keys_from_config: bool = False,
        feat_cache: Optional[dict[str, Optional[float]]] = None,
        feat_keys: Optional[Collection[str]] = None,
    ) -> None:
        """
        Initialize a BaseExtendedSequence instance.

        Parameters
        ----------
        seq : Union[str, BaseSeqRepr]
            A string represents actual amino acid sequence,

            A `BaseSeqRepr` contains the information required to build an amino
            acid sequence. If there is no mutation, the underlying sequence
            is taken. If there is a mutation, it is applied after the underlying
            sequence is taken.

        feat_config_path : Optional[Path]
            Path to the feature configuration file.
            Is ignored if either of the other two arguments are provided.
            Must be keyword argument, defaults to config.json in this directory.

        override_feat_keys_from_config : bool
            By default, `feat_cache` keys are stored into the static attribute
            `_feat_keys_from_config`. This way, if many sequences have to be
            created they can be created without opening the same file many
            times.

            However, in case the config file is changed during runtime,
            the `override_feat_keys_from_config` parameter allows `__init__`
            to reload and recache the feature keys from the new file.
            Also a keyword argument.

        feat_cache : Optional[dict[str, Optional[float]]]
            Optionally start with a partially pre-calculated set of features.
            Must be keyword argument, cannot be provided with `feat_keys`.

        feat_keys : Optional[Collection[str]]
            Optionally start with keys from another dictionary, but don't
            calculated features.
            Must be keyword argument, cannot be provided with `feat_cache`.

        Raises
        ------
        IDRDesignerException
            If more than one of the keyword arguments are provided.
        """

        if isinstance(seq, str):
            try:
                assert_valid_sequence(seq)
            except AlphabetValueException as e:
                raise IDRDesignerException(
                    "Could not initialize this sequence from string:\n"
                    + f"{pprint.pformat(seq)}\n"
                    + f"Character {e.char} was detected at index {e.i},\n"
                    + "which is not a valid amino acid.\n"
                ) from e
            self.seq = seq
        elif seq.mutation is not None:
            prev_seq: str = seq.inner.seq
            self.seq = (
                prev_seq[: seq.mutation.loc]
                + seq.mutation.end_aa
                + prev_seq[seq.mutation.loc + 1 :]
            )
        else:
            self.seq = seq.inner.seq
        if feat_keys is not None and feat_cache is not None:
            raise IDRDesignerException(
                f"Bad initialization of {type(self).__name__}.\n"
                + "Must not provide both feat_cache and feat_keys\n"
            )
        if feat_cache is not None:
            self.feat_cache = copy(feat_cache)
        elif feat_keys is not None:
            self.feat_cache = {key: None for key in feat_keys}
        else:
            self.feat_cache = self._init_cache_from_config_file(
                feat_config_path, override_feat_keys_from_config
            )
        self._set_builtin_count(None if isinstance(seq, str) else seq)

        self.procharged_res = self._init_procharged(
            None if isinstance(seq, str) else seq
        )
        self.charged_res = self._init_charged_res(
            None if isinstance(seq, str) else seq
        )

    def _init_cache_from_config_file(
        self,
        feat_config_path: Path,
        override_feat_keys_from_config: bool = False,
    ) -> dict[str, Optional[float]]:
        """
        Helper for `__init__`. Initializes feature cache keys from config file.

        Raises
        ------
        IDRDesignerException
            If there are problems opening the file or decoding the JSON.
        """
        if (
            not override_feat_keys_from_config
        ) and feat_config_path in BaseExtendedSequence._feat_keys_from_config:
            return {
                key: None
                for key in BaseExtendedSequence._feat_keys_from_config[
                    feat_config_path
                ]
            }
        result: dict[str, Optional[float]] = {}
        try:
            with open(feat_config_path, "rt", encoding="utf-8") as file:
                config: Any = json.load(file)
        except FileNotFoundError as e:
            raise IDRDesignerException(
                f"Could not open file at {feat_config_path}\n"
            ) from e
        except json.JSONDecodeError as e:
            raise IDRDesignerException(
                f"Could not decode JSON at {feat_config_path}\n"
            ) from e
        for feat_type in config:
            for feat in config[feat_type]:
                result.update([(feat, None)])
        BaseExtendedSequence._feat_keys_from_config[feat_config_path] = list(
            result.keys()
        )
        return result

    def _set_builtin_count(self, seq: "Optional[BaseSeqRepr]"):
        """Helper for `__init__`. Sets feat_cache[_X] values."""
        if seq is None:
            for aa in AMINOACIDS:
                self.feat_cache.update(
                    [(f"_{aa}", sum(1 for res in self.seq if res == aa))]
                )
        else:
            for aa in AMINOACIDS:
                val: Optional[float] = seq.inner.feat_cache[f"_{aa}"]
                if val is None:
                    raise UnreachableCode(
                        f'{type(seq).__name__} is missing field "_{aa}"'
                        + "which is always generated during initialization."
                        + f"\nseq={pprint.pformat(seq)}\n"
                    )
                if seq.mutation is not None:
                    val += int(seq.mutation.end_aa == aa)
                    val -= int(seq.mutation.start_aa == aa)
                self.feat_cache.update([(f"_{aa}", val)])

    def _init_procharged(
        self, seq_repr: "Optional[BaseSeqRepr]" = None
    ) -> list[int]:
        """
        Helper for `__init__`.
        Returns ordered positions of prolines and charged residues.
        """
        if seq_repr is None:
            return [
                i
                for i, res in enumerate(self.seq)
                if res in CHARGED_RES + ["P"]
            ]
        mut = seq_repr.mutation
        if mut is None or (
            (mut.start_aa in CHARGED_RES + ["P"])
            == (mut.end_aa in CHARGED_RES + ["P"])
        ):
            return seq_repr.inner.procharged_res
        if mut.start_aa in CHARGED_RES + ["P"]:
            return [i for i in seq_repr.inner.procharged_res if i != mut.loc]
        return sorted(seq_repr.inner.procharged_res + [mut.loc])

    def _init_charged_res(
        self, prev: Optional["BaseSeqRepr"] = None
    ) -> list[int]:
        """
        Helper for `__init__`. Returns ordered positions of charged residues.
        """
        if prev is None:
            return [i for i, res in enumerate(self.seq) if res in CHARGED_RES]
        mut = prev.mutation
        if mut is None or (
            (mut.start_aa in CHARGED_RES) == (mut.end_aa in CHARGED_RES)
        ):
            return prev.inner.charged_res
        if mut.start_aa in CHARGED_RES:
            return [i for i in prev.inner.charged_res if i != mut.loc]
        return sorted(prev.inner.charged_res + [mut.loc])

    @staticmethod
    def dict_from_fasta(fasta_path: Path) -> "dict[str, BaseExtendedSequence]":
        """
        Opens a fasta file and generates a dict with a BaseExtendedSequence
        using each fasta header as a key.

        Parameters
        ----------
        fasta_path : Path
            Path to the fasta file.

        Raises
        ------
        IDRDesignerException
            If there are problems reading the file, or
            if there are non amino acid characters in non-header lines.
        """

        str_dict: dict[str, str] = {}
        seq_name: str = ""

        try:
            with open(fasta_path, "rt", encoding="utf-8") as file:
                for n, line in enumerate(file):
                    line: str = line.strip()
                    if line.startswith(">"):
                        seq_name = line[1:]
                        str_dict[seq_name] = ""
                    else:
                        try:
                            assert_valid_sequence(line)
                        except AlphabetValueException as e:
                            raise IDRDesignerException(
                                f"Could not initialize sequence {seq_name} "
                                + f"from file {fasta_path}\n"
                                + f"Character {e.char} was detected "
                                + f"(Ln {n}, Col {e.i}, Index "
                                + f"{len(str_dict[seq_name]) + e.i})\n"
                                + "which is not a valid amino acid."
                            ) from e
                        str_dict[seq_name] += line
        except FileNotFoundError as e:
            raise IDRDesignerException(
                f"Could not open file at {fasta_path}\n"
            ) from e
        return {name: BaseExtendedSequence(s) for name, s in str_dict.items()}


class BasePointMutation:  # pylint: disable=too-few-public-methods
    """
    Represents a missense point substitution mutation.
    """

    start_aa: str
    loc: int
    """Index of mutation."""
    end_aa: str

    def __init__(self, seq: str, loc: int, end_aa: str) -> None:
        """
        Initializes a `BasePointMutation` instance.

        Parameters
        ----------
        seq : str
            The sequence to mutate.

        loc : int
            The 0-based index where the mutation will occur.

        end_aa : str
            The single amino acid character that `seq[loc]` will be.

        Raises
        ------
        IDRDesignerException
            If the mutation does nothing or if the `loc` parameter is out of
            bounds.
        """
        if not 0 <= loc < len(seq):
            raise IDRDesignerException(
                f"Invalid loc: {loc} in initialization of {type(self).__name__}"
            )
        if seq[loc] == end_aa:
            raise IDRDesignerException(
                f"Null {type(self).__name__}: start_aa = end_aa = {seq[loc]}"
            )
        self.start_aa = seq[loc]
        self.loc = loc
        self.end_aa = end_aa


class BaseSeqRepr:  # pylint: disable=too-few-public-methods
    """
    A container with a `BaseExtendedSequence` an optional mutation,
    for feature calculators to unwrap and handle.

    Based on the core idea that feature calculators can calculate faster
    if a feature value is cached for a sequence one point mutation away.
    """

    inner: BaseExtendedSequence
    mutation: Optional[BasePointMutation]
    """
    The optional point mutation that determines how a `BaseSeqRepr` instance
    should be treated.
    
    If there is no mutation, the instance is treated effectively as a
    `BaseExtendedSequence` representing the sequence which one wants to know
    the sequence features of.
    
    If there is a mutation, the `inner` attribute is treated as a
    previous sequence which is one point mutation away from the
    current sequence that one wants to know the sequence features of.
    """

    def __init__(
        self,
        seq: Union[str, BaseExtendedSequence],
        loc_end_aa: Optional[tuple[int, str]] = None,
    ) -> None:
        """
        Initialize a `BaseSeqRepr` instance.

        Parameters
        ----------
        seq : Union[str, BaseExtendedSequence]
            Current or previous sequence.

            If a previous sequence is provided (i.e. mutation ≠ None), it is
            assumed that most of the features should already be
            calculated and cached.

        loc_end_aa : Optional[tuple[int, str]]
            Location and final amino acid describing the mutation.

        Raises
        ------
        IDRDesignerException
            If the mutation does nothing or if the `loc` parameter is out of
            bounds. (from `BasePointMutation.__init__`)
        """
        self.inner = BaseExtendedSequence(seq) if isinstance(seq, str) else seq
        if loc_end_aa is None:
            self.mutation = None
        else:
            self.mutation = BasePointMutation(self.inner.seq, *loc_end_aa)

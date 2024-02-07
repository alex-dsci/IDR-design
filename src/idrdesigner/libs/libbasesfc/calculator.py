"""
A feature calculator for a base set of sequence features.

Wraps all the functionality in `libbasefeats` into a class with the
`SeqFeatCalcAPI` syntax.

Author: Tian Hao Huang (@alex-dsci)
Date: Feb 3rd, 2024
"""

import json
import pprint
from collections.abc import Collection
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from numpy import float64
from numpy._typing import NDArray

from idrdesigner.core.exceptions import IDRDesignerException
from idrdesigner.libs import assert_keys_are_subset
from idrdesigner.libs.libbasesfc import default_config_path
from idrdesigner.libs.libbasesfc.features import (
    complexity,
    count_pat,
    custom_kappa,
    custom_omega,
    fcr,
    isoelectric_point,
    length_pat,
    log_ratio,
    scd,
    score_pat,
)
from idrdesigner.libs.libbasesfc.sequences import (
    BaseExtendedSequence,
    BaseSeqRepr,
)
from idrdesigner.libs.libsfcapi import SeqFeatCalcAPI


class BaseSeqFeatCalc(SeqFeatCalcAPI[BaseSeqRepr]):  # noqa: E501, pylint: disable=too-few-public-methods, line-too-long
    """
    Base Sequence Feature Calculator

    A feature calculator for a base set of sequence features.
    See `SeqFeatCalcAPI` for how to call these features.

    The current version includes the following non-modular (hard-coded)
    sequence features:
    - Isoelectric point
    - Custom kappa, a kappa-like neighbors calculation
    - Custom omega, an omega-like neighbors calculation
    - Sequence charge decoration (scd)
    - Sequence complexity
    - Fraction of charged residues (fcr)

    As well as several modular features which can be modified via the use of
    a config.json file (see `__init__`):
    - Weighted or unweighted sums of text (regex) pattern occurrences
    - Average score calculations based on a pattern-score dictionary
    - Length spanned by text patterns
    - Log ratio between the counts of two amino acids
        (almost, use log(1 + x) to avoid funny zero behaviour)
    """

    next_seq_cache: BaseExtendedSequence
    """
    Used so that feature calculating functions can accumulate information in one
    `BaseExtendedSequence` object. The intended use case for this attribute
    is described in (TODO)
    """

    def __init__(self, feat_config_path: Path = default_config_path) -> None:
        """
        Initializes a `BaseSeqFeatCalc` instance.

        Loads in features based on functions in a configuration JSON file.

        Expected JSON structure
        -----------------------
        Non-modular features which are hard-coded and require no additional
        parameters are listed under the "non-modular" key.

        Modular features are listed under the following keys.
        Every sub-key is a feature name, the value stored at that sub-key is a
        dict containing various fields (key-value) required for the calculation
        of the feature.

        Here are the (super) keys and recognized sub-keys in the JSON config:
        - "count"
            For computing weighted counts and average scores.
            * "pat" : str
                Indicates a pattern to be counted.
                Exactly ONE of "pat" or "score" should be present.
            * "score" : str
                Stores a dictionary of patterns to their weights,
                for a weighted sum.
                Exactly ONE of "pat" or "score" should be present.
            * "average" : bool
                Optional, indicates whether the result should be divided
                by the sequence length. Defaults to `false`.
        - "length"
            For computing lengths spanned by patterns.
            * "pat" : str
                Indicates a pattern whose length spanned is to be counted.
        - "log_ratio"
            For computing log ratios between two amino acids.
            * "numerator" : str
                One letter code of amino acid in numerator of log.
            * "denominator" : str
                One letter code of amino acid in denominator of log.

        Parameters
        ----------
        feat_config_path : Path
            Path to a JSON file for feature configuration (described above)
            Defaults to config.json, in the same directory as this file.

        Raises
        ------
        IDRDesignerException
            If there are problems opening and decoding the file.

            If the provided JSON is incompatible with the above expected JSON
            structure.
        """
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

        self.features = {}

        try:
            assert_keys_are_subset(
                "Configuration JSON",
                config,
                ["non_modular", "count", "length", "log_ratio"],
            )
            if "non_modular" in config:
                self._init_non_modular_feats(config["non_modular"])
            if "count" in config:
                for name, entry in config["count"].items():
                    self._init_count_feat(name, entry)
            if "length" in config:
                for name, entry in config["length"].items():
                    self._init_length_feat(name, entry)
            if "log_ratio" in config:
                for name, entry in config["log_ratio"].items():
                    self._init_log_ratio_feat(name, entry)
        except IDRDesignerException as e:
            raise IDRDesignerException(
                f"Invalid initialization of {type(self).__name__} from "
                + f"{feat_config_path}.\n"
                + "Valid JSON but invalid format.\n"
            ) from e

    def _init_non_modular_feats(self, non_modular_feats: Collection[str]):
        """
        Helper for `__init__`.
        Deals with the "non_modular" section of config.
        """
        assert_keys_are_subset(
            "Non-modular feats",
            non_modular_feats,
            [
                "isoelectric_point",
                "custom_kappa",
                "custom_omega",
                "scd",
                "complexity",
                "fcr",
            ],
        )
        partial_inner: Callable[
            [BaseExtendedSequence, Optional[BaseSeqRepr]], float
        ]
        partial_outer: Callable[[BaseSeqRepr], float]
        if "isoelectric_point" in non_modular_feats:
            partial_outer = partial(
                self._wrap_infer_target,
                isoelectric_point,
            )
            self.features.update([("isoelectric_point", partial_outer)])
        if "custom_kappa" in non_modular_feats:
            partial_outer = partial(
                self._wrap_infer_target,
                custom_kappa,
            )
            self.features.update([("custom_kappa", partial_outer)])
        if "custom_omega" in non_modular_feats:
            partial_outer = partial(
                self._wrap_infer_target,
                custom_omega,
            )
            self.features.update([("custom_omega", partial_outer)])
        if "scd" in non_modular_feats:
            partial_outer = partial(self._wrap_infer_target, scd)
            self.features.update([("scd", partial_outer)])
        if "complexity" in non_modular_feats:
            partial_inner = partial(self._signature_convert, f=complexity)
            partial_outer = partial(self._wrap_infer_target, partial_inner)
            self.features.update([("complexity", partial_outer)])
        if "fcr" in non_modular_feats:
            partial_inner = partial(self._signature_convert, f=fcr)
            partial_outer = partial(self._wrap_infer_target, partial_inner)
            self.features.update([("fcr", partial_outer)])

    def _init_length_feat(self, name: str, entry: Any):
        """
        Helper for `__init__`.
        Deals with the "length" section of modular features config.
        """
        assert_keys_are_subset(f"Length feat {name}", entry, ["pat"])
        if "pat" not in entry.keys():
            raise IDRDesignerException(
                'Length entry should contain the "pat" field.\n'
                + f"Instead, this length entry (at {name}) was found:\n"
                + f"{pprint.saferepr(entry)}\n"
            )
        pat: str = entry["pat"]
        partial_inner: Callable[
            [BaseExtendedSequence, Optional[BaseSeqRepr]], float
        ] = partial(
            self._signature_convert,
            f=length_pat,
            fkwargs={"feat_name": name, "pattern": pat},
        )
        partial_outer: Callable[[BaseSeqRepr], float] = partial(
            self._wrap_infer_target,
            partial_inner,
        )
        self.features.update([(name, partial_outer)])

    def _init_log_ratio_feat(self, name: str, entry: Any):
        """
        Helper for `__init__`.
        Deals with the "log_ratio" section of modular features config.
        """
        assert_keys_are_subset(
            f"Log ratio feat {name}", entry, ["numerator", "denominator"]
        )
        if "numerator" not in entry.keys() or "denominator" not in entry.keys():
            raise IDRDesignerException(
                'Log ratio entry should contain the "numerator" and '
                + '"denominator" fields.\n'
                + f"Instead, this log ratio entry (at {name}) was found:\n"
                + f"{pprint.saferepr(entry)}\n"
            )
        num: str = entry["numerator"]
        denom: str = entry["denominator"]
        partial_inner: Callable[
            [BaseExtendedSequence, Optional[BaseSeqRepr]], float
        ] = partial(
            self._signature_convert,
            f=log_ratio,
            fkwargs={
                "num_name": f"_{num}",
                "num_pat": num,
                "denom_name": f"_{denom}",
                "denom_pat": denom,
                "feat_name": name,
            },
        )
        partial_outer: Callable[[BaseSeqRepr], float] = partial(
            self._wrap_infer_target,
            partial_inner,
        )
        self.features.update([(name, partial_outer)])

    def _init_count_feat(self, name: str, entry: Any):
        """
        Helper for `__init__`.
        Deals with the "count" section of modular features config.
        """
        assert_keys_are_subset(
            f"Count feat {name}", entry, ["pat", "score", "average"]
        )
        average: bool = False
        if "average" in entry:
            average = entry["average"]
        if "score" in entry and "pat" not in entry:
            dct: dict[str, float] = entry["score"]
            partial_inner: Callable[
                [BaseExtendedSequence, Optional[BaseSeqRepr]], float
            ] = partial(
                self._signature_convert,
                f=score_pat,
                fkwargs={
                    "feat_name": name,
                    "scores": dct,
                    "average": average,
                },
            )
            partial_outer: Callable[[BaseSeqRepr], float] = partial(
                self._wrap_infer_target,
                partial_inner,
            )
        elif "pat" in entry and "score" not in entry:
            pat: str = entry["pat"]
            partial_inner: Callable[
                [BaseExtendedSequence, Optional[BaseSeqRepr]], float
            ] = partial(
                self._signature_convert,
                f=count_pat,
                fkwargs={
                    "feat_name": name,
                    "pattern": pat,
                    "average": average,
                },
            )
            partial_outer: Callable[[BaseSeqRepr], float] = partial(
                self._wrap_infer_target,
                partial_inner,
            )
        else:
            raise IDRDesignerException(
                'Count entry should contain the "score" field OR "pat" '
                + "field (mutually exclusive).\n"
                + f"Instead, this count entry (at {name}) was found:\n"
                + f"{pprint.saferepr(entry)}\n"
            )
        self.features.update([(name, partial_outer)])

    def _wrap_infer_target(
        self,
        f: Callable[
            [
                BaseExtendedSequence,
                Optional[BaseSeqRepr],
            ],
            float,
        ],
        seq: BaseSeqRepr,
    ) -> float:
        """
        A DRY-compliant abstraction of using `next_seq_cache` as the target
        if the inputted sequence contains a point mutation, otherwise using
        the inputted sequence as a target. No `*args` or `**kwargs` suppport
        provided.

        Parameters
        ----------
        f : Callable[[BaseExtendedSequence, Optional[BaseSeqRepr]], float]
            A function which takes a target sequence and an optional
            previous seq and mutation for context.

        seq : BaseSeqRepr
            Represents the target sequence if it has no mutation information
            but otherwise represents the last (one point mut away) sequence.
        """
        target: BaseExtendedSequence = (
            self.next_seq_cache if seq.mutation is not None else seq.inner
        )
        prev: Optional[BaseSeqRepr] = seq if seq.mutation is not None else None
        return f(target, prev)

    def _signature_convert(
        self,
        target: BaseExtendedSequence,
        _prev: Any,
        f: Callable[..., float],
        fargs: Optional[list[Any]] = None,
        fkwargs: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        A DRY-compliant way of turning functions which take no previous mutation
        context into ones which are compatible with `_wrap_infer_target`.
        Used with `partial`. Supports `*args` and `**kwargs`.

        Parameters
        ----------
        target : BaseExtendedSequence
            Target sequence from which either the cached value will be retrieved
            or a new calculation will be performed.

        _prev : (dropped)
            A dummy variable to match the signature of functions accepted by
            `_wrap_infer_target`.

        f : Callable[[str, ...], float]
            A function which takes no previous sequence representation.

            Must take a string as first argument, and then other arguments can
            be customized based on compatibility with fargs and fkwargs.

        fargs: Optional[list[Any]]
            Optional list of positional arguments the function is going to need.
            For similar use cases as partial.

        fkwargs: Optional[dict[str, Any]]
            Optional dict of keyword arguments the function is going to need.
            For similar use cases as partial.
        """
        return f(target, *(fargs or []), **(fkwargs or {}))

    def __call__(
        self, _seq: BaseSeqRepr, target_feats: Optional[Collection[str]] = None
    ) -> NDArray[float64]:
        """
        Wraps the call to `super().__call__` by setting a new
        `BaseExtendedSequence` in the sequence cache if necessary.

        If the provided `_seq` has:
            No mutation => the `next_seq_cache` will be ignored as all
            calculations can be cached on `_seq`, which represents the
            current sequence of interest.

            A mutation => a sequence at the `next_seq_cache` will be treated
            as the target and feature values will be cached there.
        """
        if _seq.mutation is not None:
            self.next_seq_cache = BaseExtendedSequence(
                _seq, feat_keys=self.features
            )
        return super().__call__(_seq, target_feats)

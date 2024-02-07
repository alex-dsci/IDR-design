"""Tests for libs.libbasesfc.sequences"""
from typing import Any, Optional, Generator, Collection
from copy import copy
from pathlib import Path
import json
import shutil
from tempfile import TemporaryDirectory
from functools import partial
import pytest
from idrdesigner.core.consts import CHARGED_RES, AMINOACIDS
from idrdesigner.core.exceptions import IDRDesignerException
from idrdesigner.libs import AlphabetValueException
from idrdesigner.libs.libbasesfc.sequences import (
    BaseExtendedSequence,
    BasePointMutation,
    BaseSeqRepr,
)


from . import (
    all_feats_cache,
    small_feats_cache,
    small_config_file_path,
    plain_text_path,
    data_path,
    assert_fails,
)


def generate_tests_BES_init() -> (  # pylint: disable=invalid-name
    Generator[
        tuple[
            list[Any], dict[str, Any], str, Optional[dict[str, Optional[float]]]
        ],
        None,
        None,
    ]
):
    """Generates test cases for `test_BES_init`."""
    local_cache1: dict[str, Optional[float]] = copy(all_feats_cache)
    local_cache1.update(
        {
            "_M": 1,
            "_E": 2,
            "_T": 1,
            "_R": 1,
        }
    )
    # >>> BaseExtendedSequence("METER")
    # Testing to see it has all the base feats and counts for M, E, T, and R.
    yield ["METER"], {}, "METER", local_cache1
    # >>> BaseExtendedSequence(
    # >>>     "METER", feat_cache={..., "_M": 1, "MOD_CDK_SPxK_1": None}
    # >>> )
    # Testing to see it can initialize with a feature cache.
    yield (
        ["METER"],
        {"feat_cache": copy(local_cache1)},
        "METER",
        copy(local_cache1),
    )
    local_cache2: dict[str, Optional[float]] = copy(local_cache1)
    local_cache2.update({"_M": None})
    # >>> BaseExtendedSequence(
    # >>>     "METER", feat_cache={..., "_M": None, "MOD_CDK_SPxK_1": None}
    # >>> )
    # Testing to see _X (amino acid counts) are set privately.
    yield (
        ["METER"],
        {"feat_cache": copy(local_cache2)},
        "METER",
        copy(local_cache1),
    )
    local_cache2.pop("MOD_CDK_SPxK_1")
    local_cache3: dict[str, Optional[float]] = copy(local_cache2)
    local_cache3.pop("_M")
    local_cache2.update({"_M": 1})
    # >>> BaseExtendedSequence(
    # >>>     "METER", feat_cache={...}
    # >>> )
    # Testing to see any non-private (_X) feature can be removed
    # from the feature set.
    yield (
        ["METER"],
        {"feat_cache": copy(local_cache3)},
        "METER",
        copy(local_cache2),
    )
    local_cache1: dict[str, Optional[float]] = copy(small_feats_cache)
    local_cache1.update(
        {
            "_M": 1,
            "_E": 2,
            "_T": 1,
            "_R": 1,
        }
    )
    # >>> BaseExtendedSequence(
    # >>>     "METER",
    # >>>     feat_config_path=small_config_file_path
    # >>> )
    # Testing to see initialization based on non-default config file
    # with non-default modular features.
    yield (
        ["METER"],
        {"feat_config_path": small_config_file_path},
        "METER",
        copy(local_cache1),
    )
    # >>> BaseExtendedSequence("METER", feat_cache={..., "A_minus_G": None})
    # Testing to see initialization based on non-default feature cache
    # overrides the default config file path.
    yield (
        ["METER"],
        {"feat_cache": copy(local_cache1)},
        "METER",
        copy(local_cache1),
    )
    # >>> BaseExtendedSequence(
    # >>>     "METER",
    # >>>     feat_keys={..., "A_minus_G": None}.keys()
    # >>> )
    # Testing to see initialization based on non-default feature keys,
    # and checking that it overrides the default config file path.
    yield (
        ["METER"],
        {"feat_keys": list(local_cache1.keys())},
        "METER",
        copy(local_cache1),
    )

    # In order to test the constructor with `BaseSeqRepr` instances,
    # we assume the `BaseSeqRepr` tests are working, which assumes that
    # the previous yield cases in this function are also working.

    # >>> BaseExtendedSequence(
    # >>>     BaseSeqRepr("METER")
    # >>> )
    yield ([BaseSeqRepr("METER")], {}, "METER", None)
    # >>> BaseExtendedSequence(
    # >>>     BaseSeqRepr("METER", (0, "P"))
    # >>> )
    # Should output a base extended sequence with the mutation applied; i.e.
    # should be "PETER".
    # Also, for code coverage, mutate an amino acid to a proline.
    yield ([BaseSeqRepr("METER", (0, "P"))], {}, "PETER", None)
    # For code coverage, mutate a charged amino acid to a non-charged one.
    yield ([BaseSeqRepr("METER", (1, "A"))], {}, "MATER", None)
    # For code coverage, mutate a non-charged amino acid to a charged one.
    yield ([BaseSeqRepr("METER", (2, "E"))], {}, "MEEER", None)


@pytest.mark.parametrize(
    "args,kwargs,seq,feat_cache",
    generate_tests_BES_init(),
)
def test_BES_init(  # pylint: disable=invalid-name
    args: list[Any],
    kwargs: dict[str, Any],
    seq: str,
    feat_cache: Optional[dict[str, Optional[float]]],
):
    """
    Test that `BaseExtendedSequence.__init__` produces a sensible output.
    If `feat_cache` is not provided, assume it is fine.
    """
    seq_repr: BaseExtendedSequence = BaseExtendedSequence(*args, **kwargs)
    assert seq_repr.seq == seq
    if feat_cache is not None:
        assert seq_repr.feat_cache == feat_cache
    assert sorted(seq_repr.charged_res) == seq_repr.charged_res
    assert all(seq_repr.seq[loc] in CHARGED_RES for loc in seq_repr.charged_res)
    assert not any(
        res in CHARGED_RES
        for loc, res in enumerate(seq_repr.seq)
        if loc not in seq_repr.charged_res
    )
    assert sorted(seq_repr.procharged_res) == seq_repr.procharged_res
    assert all(
        seq_repr.seq[loc] in CHARGED_RES + ["P"]
        for loc in seq_repr.procharged_res
    )
    assert not any(
        res in CHARGED_RES + ["P"]
        for loc, res in enumerate(seq_repr.seq)
        if loc not in seq_repr.procharged_res
    )


def generate_tests_BES_init_fails() -> (  # pylint: disable=invalid-name
    Generator[
        tuple[list[Any], dict[str, Any], list[type]],
        None,
        None,
    ]
):
    """Generates test cases for `test_BES_init_fails`."""
    # >>> BaseExtendedSequence("NOT AN AMINO ACID SEQUENCE")
    # Test that non amino-acid characters are caught.
    yield (
        ["NOT AN AMINO ACID SEQUENCE"],
        {},
        [IDRDesignerException, AlphabetValueException],
    )
    # >>> BaseExtendedSequence("Meter")
    # Test that lower-case characters are caught.
    yield (
        ["Meter"],
        {},
        [IDRDesignerException, AlphabetValueException],
    )
    local_cache1: dict[str, Optional[float]] = copy(all_feats_cache)
    local_cache1.update(
        {
            "_M": 1,
            "_E": 2,
            "_T": 1,
            "_R": 1,
        }
    )
    # >>> BaseExtendedSequence("METER", feat_keys=[...], feat_cache={...})
    # Test that feat_keys and feat_cache can't be provided simultaneously.
    yield (
        ["METER"],
        {
            "feat_keys": list(local_cache1.keys()),
            "feat_cache": copy(local_cache1),
        },
        [IDRDesignerException],
    )
    # >>> BaseExtendedSequence(
    # >>>     "METER",
    # >>>     feat_config_path="non_existent_file.json",
    # >>> )
    # Test that FileNotFound is wrapped in an IDRDesignerException.
    yield (
        ["METER"],
        {"feat_config_path": Path("non_existent_file.json")},
        [IDRDesignerException, FileNotFoundError],
    )
    # >>> BaseExtendedSequence(
    # >>>     "METER",
    # >>>     feat_config_path=plain_text_path,
    # >>> )
    # Test that FileNotFound is wrapped in an IDRDesignerException.
    yield (
        ["METER"],
        {"feat_config_path": plain_text_path},
        [IDRDesignerException, json.JSONDecodeError],
    )


@pytest.mark.parametrize(
    "args,kwargs,errors",
    generate_tests_BES_init_fails(),
)
def test_BES_init_fails(  # pylint: disable=invalid-name
    args: list[Any], kwargs: dict[str, Any], errors: list[type]
):
    """Test that `BaseExtendedSequence.__init__` fails with bad inputs."""
    f = partial(BaseExtendedSequence, *args, **kwargs)
    assert_fails(f, errors)


@pytest.mark.parametrize(
    "kwargs,feat_keys",
    [
        # >>> BaseExtendedSequence(
        # >>>     "METER",
        # >>>     feat_config_path="...temp.json",
        # >>> )
        (
            {},
            [
                "scd",
                "fcr",
                "cold_regex_pattern",
                "A_minus_G",
                "percent_P_or_T",
                "cold_regex_length",
                "ED_ratio",
            ]
            + [f"_{aa}" for aa in AMINOACIDS],
        ),
        # >>> BaseExtendedSequence(
        # >>>     "METER",
        # >>>     feat_config_path="...temp.json",
        # >>>     override_feat_keys_from_config=True,
        # >>> )
        (
            {"override_feat_keys_from_config": True},
            [
                "cold_regex_pattern",
                "A_minus_G",
                "percent_P_or_T",
                "cold_regex_length",
                "ED_ratio",
            ]
            + [f"_{aa}" for aa in AMINOACIDS],
        ),
    ],
)
def test_BES_init_override_feat_keys_from_config(
    kwargs: dict[str, Any], feat_keys: Collection[str]
):  # pylint: disable=invalid-name
    """
    Test that `BaseExtendedSequence.__init__` can respond to config file
    updates using the `override_feat_keys_from_config` parameter.
    """
    with TemporaryDirectory() as temp_dir:
        temp_json_path = Path(temp_dir).joinpath("temp.json")
        with open(temp_json_path, "wt", encoding="utf-8") as temp_file, open(
            small_config_file_path, "rt", encoding="utf-8"
        ) as file:
            shutil.copyfileobj(file, temp_file)
            file.seek(0)
            config: Any = json.load(file)
        _ = BaseExtendedSequence(
            "METER", feat_config_path=temp_json_path, **kwargs
        )
        config.pop("non_modular")
        with open(temp_json_path, "wt", encoding="utf-8") as temp_file:
            json.dump(config, temp_file)
        seq: BaseExtendedSequence = BaseExtendedSequence(
            "METER", feat_config_path=temp_json_path, **kwargs
        )
    assert list(seq.feat_cache.keys()) == feat_keys


def test_BES_dict_from_fasta():  # pylint: disable=invalid-name
    """Test that `BaseExtendedSequence.dict_from_fasta` runs."""
    BaseExtendedSequence.dict_from_fasta(data_path.joinpath("tiny.fasta"))


@pytest.mark.parametrize(
    "fasta_path,errors",
    [
        # >>> BaseExtendedSequence.dict_from_fasta("...tiny_bad.fasta")
        # Test non amino-acid characters caught.
        (
            data_path.joinpath("tiny_bad.fasta"),
            [IDRDesignerException, AlphabetValueException],
        ),
        # >>> BaseExtendedSequence.dict_from_fasta("non_existent_file.fasta")
        # Test file not found errors wrapped in IDRDesignerException
        (
            "non_existent_file.fasta",
            [IDRDesignerException, FileNotFoundError],
        ),
    ],
)
def test_BES_dict_from_fasta_fails(  # pylint: disable=invalid-name
    fasta_path: Path, errors: list[type]
):
    """Test that `BaseExtendedSequence.dict_from_fasta` fails on bad inputs."""
    f = partial(BaseExtendedSequence.dict_from_fasta, fasta_path)
    assert_fails(f, errors)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        # >>> BasePointMutation("METER", 0, "E")
        # Make sure indexing is 0-based. Will fail if indexing is 1-based.
        (["METER", 0, "E"], {}),
        # >>> BasePointMutation(seq="METER", loc=0, end_aa="E")
        # kwargs version
        ([], {"seq": "METER", "loc": 0, "end_aa": "E"}),
    ],
)
def test_BPM_init(  # pylint: disable=invalid-name
    args: list[Any], kwargs: dict[str, Any]
):
    """Test that `BasePointMutation.__init__` generates sensible output."""
    mutation: BasePointMutation = BasePointMutation(*args, **kwargs)
    seq: str = args[0] if len(args) > 0 else kwargs["seq"]
    assert mutation.start_aa != mutation.end_aa
    assert seq[mutation.loc] == mutation.start_aa
    assert 0 <= mutation.loc < len(seq)


@pytest.mark.parametrize(
    "args,kwargs,errors",
    [
        # >>> BasePointMutation("METER", 2, "T")
        # Test catches that we are mutating T to T, which is bad.
        (["METER", 2, "T"], {}, [IDRDesignerException]),
        # >>> BasePointMutation("METER", -1, "E")
        # Test fails because index out of bounds.
        (["METER", -1, "E"], {}, [IDRDesignerException]),
        # >>> BasePointMutation("METER", 5, "E")
        # Test fails because index out of bounds.
        (["METER", 5, "E"], {}, [IDRDesignerException]),
    ],
)
def test_BPM_init_fails(  # pylint: disable=invalid-name
    args: list[Any], kwargs: dict[str, Any], errors: list[type]
):
    """Test that `BasePointMutation.__init__` fails on bad input."""
    f = partial(BasePointMutation, *args, **kwargs)
    assert_fails(f, errors)


@pytest.mark.parametrize(
    "args,kwargs,seq,loc_end_aa",
    [
        # >>> BaseSeqRepr("METER")
        (["METER"], {}, "METER", None),
        # >>> BaseSeqRepr("METER", (2, "E"))
        (["METER", (2, "E")], {}, "METER", (2, "E")),
        # >>> BaseSeqRepr(seq="METER", loc_end_aa=(2, "E"))
        ([], {"seq": "METER", "loc_end_aa": (2, "E")}, "METER", (2, "E")),
        # >>> BaseSeqRepr(BaseExtendedSequence("METER"))
        ([BaseExtendedSequence("METER")], {}, "METER", None),
    ],
)
def test_BSR_init(  # pylint: disable=invalid-name
    args: list[Any],
    kwargs: dict[str, Any],
    seq: str,
    loc_end_aa: Optional[tuple[int, str]],
):
    """Tests that `BaseSeqRepr.__init__` produces a sensible output."""
    seq_repr = BaseSeqRepr(*args, **kwargs)
    assert seq_repr.inner.seq == seq
    if loc_end_aa is not None:
        loc, end_aa = loc_end_aa
        assert seq_repr.mutation is not None
        assert seq_repr.mutation.loc == loc
        assert seq_repr.mutation.end_aa == end_aa

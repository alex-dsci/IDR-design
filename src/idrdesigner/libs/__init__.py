"""Shared libraries for the project."""
from typing import Any, Optional, Collection

from idrdesigner.core.exceptions import IDRDesignerException
from idrdesigner.core.consts import AMINOACIDS


class AlphabetValueException(IDRDesignerException):
    """
    An exception for `assert_valid_sequence` which returns the index and
    offending character.

    Should be handled to convert the data to a nicer format.
    """

    i: int
    char: str

    def __init__(self, i: int, char: str) -> None:
        self.i = i
        self.char = char


def assert_valid_sequence(
    s: str, alphabet: Optional[Collection[str]] = None
) -> None:
    """
    Checks if a string `s` is built up of characters from an `alphabet`,
    defaulting to the standard amino acid alphabet if `None` is provided.

    Parameters
    ----------
    s : str
        The string to analyze.

    alphabet : list[str], default=`idrdesigner.core.consts.AMINOACIDS`
        A list containing all valid characters of an alphabet.

    Raises
    ------
    AlphabetValueException

    """
    _alphabet: Collection[str] = (
        alphabet if alphabet is not None else AMINOACIDS
    )

    bad_char: Optional[tuple[int, str]] = next(
        ((i, aa) for i, aa in enumerate(s) if aa not in _alphabet), None
    )
    if bad_char is not None:
        raise AlphabetValueException(*bad_char)


def assert_keys_are_subset(
    json_name: str, test_keys: Any, allowed_keys: Collection[str]
):
    """
    A DRY-compliant way of asserting the keys of a json object should be
    from a specific set.

    Parameters
    ----------
    json_name : str
        The name of the json object to help specify the error context.

    test_keys : Iterable[str]
        A json object (iterator should yield keys)

    allowed_keys : Collection[str]
        The allowed set of keys.

    Raises
    ------
    IDRDesignerException
    """
    i_bad_name: Optional[tuple[int, str]] = next(
        (
            (i, name)
            for i, name in enumerate(test_keys)
            if name not in allowed_keys
        ),
        None,
    )
    if i_bad_name is not None:
        i, bad_name = i_bad_name
        raise IDRDesignerException(
            f'During the parsing of "{json_name}", '
            + f"got unexpected key in (index {i}): {bad_name}\n"
        )

"""
Sequence feature calculator API / abstract class.
Provides common syntax for managing sequence feature calculations.

Author: Tian Hao Huang (@alex-dsci)
Date: Jan 25th, 2024
"""

import pprint
from typing import Callable, TypeVar, Generic, Optional, Any
from collections.abc import Collection
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from idrdesigner.core.exceptions import IDRDesignerException

_S = TypeVar("_S")


class SeqFeatCalcAPI(ABC, Generic[_S]):
    """
    Provides common syntax for managing sequence feature calculations.
    """

    features: dict[str, Callable[[_S], float]]

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Abstract method for initializing self.features dynamically."""
        raise NotImplementedError(
            f"{type(self).__name__} implements SeqFeatCalcAPI "
            + "but does not implement __init__!\n"
        )

    def __getitem__(self, __key: str) -> Callable[[_S], float]:
        """
        Interprets the following syntax to run specific feature calculations:
        >>> calculator = MySeqFeatCalc(...)
        >>> seq = 'MSEQQAR...'
        >>> calculator['net-charge'](seq)
        _net_charge(seq)
        """
        return self.features.__getitem__(__key)

    def __call__(
        self, _seq: _S, target_feats: Optional[Collection[str]] = None
    ) -> npt.NDArray[np.float64]:
        """
        Interprets the following syntax to run feature calculations:
        >>> calculator = MySeqFeatCalc(...)
        >>> seq = 'MSEQQAR...'
        >>> calculator(seq)
        np.array([_net_charge(seq), _hydropathy(seq), ...])
        >>> calculator(seq, ['hydrophobicity', 'net-charge', ...])
        np.array([_hydropathy(seq), _net_charge(seq), ...])
        """
        if target_feats is None:
            target_feats = self.features.keys()
        else:
            i_bad_key: Optional[tuple[int, str]] = next(
                (
                    (i, key)
                    for i, key in enumerate(target_feats)
                    if key not in self.features.keys()
                ),
                None,
            )
            if i_bad_key is not None:
                i: int
                bad_key: str
                i, bad_key = i_bad_key
                raise IDRDesignerException(
                    f"Tried to call {type(self).__name__}[{bad_key}] because "
                    + f"of call to {type(self).__name__} with the following "
                    + "target_feats:\n"
                    + f"{pprint.pformat(target_feats)}\n"
                    + f"{bad_key} at index {i} is not a feature of the class "
                    + f"{type(self).__name__}\n"
                )
        # Assumes self._fn(_seq) always returns or raises.
        # In particular, it never hangs...
        # Probably want to change this for a modular package taking
        #   user-defined functions, which can be fishy
        return np.array(
            [self[feature](_seq) for feature in target_feats], dtype=np.float64
        )

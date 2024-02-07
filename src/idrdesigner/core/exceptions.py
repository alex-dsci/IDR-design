"""Generic but not necessarily abstract exception types for the project."""


class IDRDesignerException(Exception):
    """
    IDR Designer miscellaneous exception. Used to distinguish all exceptions
    raised and handled in the IDR Design code base from those raised in external
    code.

    While all exceptions are derived from this type, a generic
    `IDRDesignerException` should only be raised when a forseeable exception
    occurs, and both of the below are true of the exception:
    - Generally doesn't occur commonly enough to be its own subtype.
    - Should be displayed, not handled
        (i.e. human-readable message payload only).
    """


class UnreachableCode(IDRDesignerException):
    """
    A fatal exception to satisfy the type checker, and for annotating program
    logic. Like AssertionError, but derives from IDRDesignerException.

    Used to assert code paths that should be logically impossible,
    or signal invalid function arguments passed to a private function.
    Should never be excepted for any reason, even in testing!
    """

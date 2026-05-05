"""
Custom exceptions raised by the package.
"""


class NotConvergingError(ValueError):
    """
    Signal that an iterative algorithm exceeded its allowed iteration budget.

    :ivar alpha: Most recent iterate available at the time the exception was
        raised.
    """

    def __init__(self, message: str, alpha: float) -> None:
        """
        Initialize the exception.

        :param message: Human-readable error message.
        :param alpha: Most recent iterate computed by the algorithm.
        :return: ``None``.
        """
        super().__init__(message)
        self.alpha = alpha

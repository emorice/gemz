"""
Interface for case output handler
"""

from abc import ABC, abstractmethod

class Output(ABC):
    """
    Interface for case output handler
    """

    @abstractmethod
    def add_title(self, title: str):
        """
        Add title. Should be called at most once and first.
        """

    @abstractmethod
    def add_figure(self, figure):
        """
        Add a figure to the output
        """

    def add_figures(self, figures):
        for fig in figures:
            self.add_figure(fig)

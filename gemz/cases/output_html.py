"""
Generates html report from case
"""

from typing import TextIO
from html import escape

from gemz.cases import Output
from gemz.reporting import write_fig

class HtmlOutput(Output):
    """
    Generates html report from case
    """
    def __init__(self, path: str):
        self.path = path
        self._stream : TextIO | None = None

    @property
    def stream(self) -> TextIO:
        """
        Type safe accesor for stream
        """
        if self._stream is None:
            raise RuntimeError('Output handler not entered yet')
        return self._stream

    def __enter__(self) -> 'HtmlOutput':
        self._stream = open(self.path, 'w', encoding='utf8')
        return self

    def __exit__(self, *args) -> None:
        self.stream.close()
        self._stream = None

    def add_figure(self, figure) -> None:
        """
        Add a figure to the output
        """
        write_fig(self.stream, figure)

    def add_title(self, title: str) -> None:
        """
        Add title. Should be called at most once and first.
        """
        self.stream.write(f'<h1>{escape(title)}</h1>')

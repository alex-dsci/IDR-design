from typing import TextIO

class ProgressLogger:
    file: TextIO
    display_mode: bool
    delimiter: str
    col_names: list[str]
    max_lens: list[int] | None
    def __init__(self, file: TextIO, col_names: list[str], display_mode: bool = False, max_lens: list[int] | None = None, delimiter: str = "\t") -> None:
        self.file = file
        self.display_mode = display_mode
        self.delimiter = delimiter
        self.col_names = col_names
        self.max_lens = max_lens
        if max_lens is not None:
            assert len(max_lens) == len(self.col_names)
    def write_header(self):
        self.write_data(self.col_names)
        if self.display_mode:
            print(file=self.file)
    def write_data(self, data: list[str]):
        if self.max_lens is None:
            self._write_no_columns(data)
        else:
            self._write_w_columns(data)
    def print(self, s: str = "", **kwargs):
        if self.display_mode:
            self.file.write("\r")
            print(s, end="", file=self.file, **kwargs)
        else:
            print(s, file=self.file, **kwargs)
    def _truncate_after_n(self, s: str, n: int) -> str:
        if len(s) > n:
            return s[:n-3] + "..."
        return s
    def _write_w_columns(self, data: list[str]):
        assert self.max_lens is not None 
        assert len(data) == len(self.max_lens)
        output = []
        for field, max_len in zip(data, self.max_lens):
            field += " " * max(0, max_len - len(field))
            output.append(self._truncate_after_n(field, max_len))
        self.print(self.delimiter.join(output)) 
    def _write_no_columns(self, data: list[str]):
        assert self.max_lens is None
        self.print(self.delimiter.join(data)) 

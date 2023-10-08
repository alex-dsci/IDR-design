from typing import TextIO, Any, cast
from abc import ABC, abstractmethod
from pandas import DataFrame, Series, concat
from shutil import get_terminal_size
from IPython.core.getipython import get_ipython
from sys import stdout

class PrintDesignProgress(ABC):
    bypass: bool
    @abstractmethod
    def enter_design_similar(self, job_name: str, tgt_seq: str, num_designs: int, algorithm: str) -> None:
        if self.bypass:
            return
    @abstractmethod
    def exit_design_similar(self, job_name: str, query_seqs: Series, final_seqs: Series, times: Series) -> None:
        if self.bypass:
            return
        qseqs = cast("Series[str]", query_seqs)
        fseqs = cast("Series[str]", final_seqs)
        ts = cast("Series[float]", times)
    @abstractmethod
    def report_round(self, guess_seq: str, iteration: int, distance: float, time: float) -> None:
        if self.bypass:
            return
    @abstractmethod
    def enter_search_similar(self, job_name: str, job_num: int, query_seq: str, distance: float) -> None:
        if self.bypass:
            return
    @abstractmethod
    def exit_search_similar(self, job_name: str, job_num: int, final_seq: str, final_distance: float, time: float) -> None:
        if self.bypass:
            return

class LogToFile(PrintDesignProgress):
    file: TextIO
    def __init__(self, file: TextIO) -> None:
        super().__init__()
        self.file = file
    def enter_design_similar(self, job_name: str, tgt_seq: str, num_designs: int, algorithm: str) -> None:
        if self.bypass:
            return
        print(f"<SEARCHES_START>", file=self.file)
        print(f"{job_name}", file=self.file)
        print(f"Algorithm, number of designs: {algorithm}, {num_designs}", file=self.file)
        print(f"Target sequence:", file=self.file)
        print(f"{tgt_seq}", file=self.file)
    def exit_design_similar(self, job_name: str, query_seqs: Series, final_seqs: Series, times: Series) -> None:
        if self.bypass:
            return
        qseqs = cast("Series[str]", query_seqs)
        fseqs = cast("Series[str]", final_seqs)
        ts = cast("Series[float]", times)
        print(f"<SEARCHES_END>", file=self.file)
        print(f"{job_name}", file=self.file)
        print(f"Average time: {sum(ts) / len(query_seqs)}", file=self.file)
        print(f"Name, time, query sequence, designed sequence:", file=self.file)
        for i, data in enumerate(zip(ts, qseqs, fseqs)):
            print(f"{job_name} {i}, {data[0]}, {data[1]}, {data[2]}", file=self.file)
    def report_round(self, guess_seq: str, iteration: int, distance: float, time: float) -> None:
        if self.bypass:
            return
        print(f"{iteration}, {guess_seq}, {distance}, {time}", file=self.file)
    def enter_search_similar(self, job_name: str, job_num: int, query_seq: str, distance: float) -> None:
        if self.bypass:
            return
        print(f"<SEARCH_START>", file=self.file)
        print(f"{job_name} {job_num}", file=self.file)
        print(f"Query sequence:", file=self.file)
        print(f"{query_seq}", file=self.file)
        print(f"Distance: {distance}", file=self.file)
        print(f"#" * len(f"Distance: {distance}"), file=self.file)
        print(f"Iteration, sequence, distance, time", file=self.file)
    def exit_search_similar(self, job_name: str, job_num: int, final_seq: str, final_distance: float, time: float) -> None:
        if self.bypass:
            return
        print(f"<SEARCH_END>", file=self.file)
        print(f"{job_name} {job_num}", file=self.file)
        print(f"Final sequence:", file=self.file)
        print(f"{final_seq}", file=self.file)
        print(f"Distance, time:",file=self.file)
        print(f"{final_distance}, {time}", file=self.file)

def _fixed_len_str(s: str, n: int) -> str:
    if len(s) > n:
        return s[:n-3] + "..."
    else:
        return s + " " * (n - len(s))
    
def _in_columns(data: list[Any], col_lens: list[int]) -> str:
    assert len(data) == len(col_lens)
    return "\t".join(map(_fixed_len_str, map(str, data), col_lens))

BUFFER = 20
FLOAT_LEN = 20
INT_LEN = 10 
NOTEBOOK_LEN = 200
def _max_line_length() -> int:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        return NOTEBOOK_LEN   # Jupyter notebook or qtconsole
    else:
        return get_terminal_size().columns  # Terminal running python
class DisplayToStdout(PrintDesignProgress):
    same_line: bool
    def __init__(self, same_line: bool = True) -> None:
        super().__init__()
        self.same_line = same_line
    def enter_design_similar(self, job_name: str, tgt_seq: str, num_designs: int, algorithm: str) -> None:
        if self.bypass:
            return
        print(f"Starting {job_name} ({num_designs} sequences). Running {algorithm}.")
        print(f"Target sequence:")
        print(f"{tgt_seq}")
    def exit_design_similar(self, job_name: str, query_seqs: Series, final_seqs: Series, times: Series) -> None:
        if self.bypass:
            return
        qseqs = cast("Series[str]", query_seqs)
        fseqs = cast("Series[str]", final_seqs)
        ts = cast("Series[float]", times)
        print(f"Finished {job_name} averaging {sum(ts) / len(qseqs)} seconds.")
        column_lengths: list[int] = [len(job_name) + len(str(len(qseqs))) + 1, FLOAT_LEN]
        remaining_length = _max_line_length() - sum(column_lengths) - BUFFER
        column_lengths = column_lengths + [int(remaining_length/2), int(remaining_length/2)]
        print(_in_columns(["Name", "time", "query sequence", "designed sequence"], column_lengths))
        for i, data in enumerate(zip(ts, qseqs, fseqs)):
            print(_in_columns([f"{job_name} {i}", data[0], data[1], data[2]], column_lengths))
    def report_round(self, guess_seq: str, iteration: int, distance: float, time: float) -> None:
        if self.bypass:
            return
        column_lengths: list[int] = [FLOAT_LEN, INT_LEN, FLOAT_LEN]
        remaining_length = _max_line_length() - sum(column_lengths) - BUFFER
        column_lengths = [remaining_length] + column_lengths
        if self.same_line:
            print("\r" + _in_columns([guess_seq, distance, iteration, time], column_lengths), end="")
            stdout.flush()
        else:
            print(_in_columns([guess_seq, distance, iteration, time], column_lengths))
    def enter_search_similar(self, job_name: str, job_num: int, query_seq: str, distance: float) -> None:
        if self.bypass:
            return
        print(f"Starting {job_name} (Query #{job_num}).")
        print(f"Query sequence:")
        remaining_length = _max_line_length() - BUFFER
        print(_fixed_len_str(query_seq, remaining_length))
        print(f"Distance:")
        print(f"{distance}")
        print(f"#" * len(str(distance)))
        column_lengths: list[int] = [FLOAT_LEN, INT_LEN, FLOAT_LEN]
        remaining_length = _max_line_length() - sum(column_lengths) - BUFFER
        column_lengths = [remaining_length] + column_lengths
        print(_in_columns(["Sequence", "distance", "iteration", "time"], column_lengths))
    def exit_search_similar(self, job_name: str, job_num: int, final_seq: str, final_distance: float, time: float) -> None:
        if self.bypass:
            return
        print(f"\nFinishing {job_name} (Query #{job_num})") 
        print(f"Final sequence:")
        remaining_length = _max_line_length() - BUFFER
        print(_fixed_len_str(final_seq, remaining_length))
        print(f"Distance, time:")
        print(f"{final_distance}, {time}")
    
SUMMARY_COLUMNS: list[str] = [
    "num",
    "time",
    "target",
    "query",
    "design",
    "design_dist"
]
ONE_DES_COLUMNS: list[str] = [
    "iteration",
    "seq",
    "dist",
    "time"
]
class LogToCSV(PrintDesignProgress):
    dir: str
    tgt: str
    qry: str
    summary_df: DataFrame
    one_design_df: DataFrame
    def __init__(self, path: str) -> None:
        super().__init__()
        self.dir = path
    def enter_design_similar(self, job_name: str, tgt_seq: str, num_designs: int, algorithm: str) -> None:
        if self.bypass:
            return
        self.summary_df = DataFrame(columns=SUMMARY_COLUMNS)
        self.tgt = tgt_seq
    def exit_design_similar(self, job_name: str, query_seqs: Series, final_seqs: Series, times: Series) -> None:
        if self.bypass:
            return
        path = f"{self.dir}/{job_name}.csv"
        self.summary_df = self.summary_df.set_index("num")
        self.summary_df.to_csv(path)
    def report_round(self, guess_seq: str, iteration: int, distance: float, time: float) -> None:
        if self.bypass:
            return
        next_row = DataFrame({
            "iteration": [iteration],
            "seq": [guess_seq],
            "dist": [distance],
            "time": [time]
        })
        if len(self.one_design_df) == 0:
            self.one_design_df = next_row
        else:     
            self.one_design_df = concat((self.one_design_df, next_row))
    def enter_search_similar(self, job_name: str, job_num: int, query_seq: str, distance: float) -> None:
        if self.bypass:
            return
        self.one_design_df = DataFrame(columns=ONE_DES_COLUMNS)
        self.qry = query_seq
    def exit_search_similar(self, job_name: str, job_num: int, final_seq: str, final_distance: float, time: float) -> None:
        if self.bypass:
            return
        path = f"{self.dir}/{job_name}_{job_num}.csv"
        self.one_design_df = self.one_design_df.set_index("iteration")
        self.one_design_df.to_csv(path)
        next_row = DataFrame({
            "num": [job_num],
            "time": [time],
            "target": [self.tgt],
            "query": [self.qry],
            "design": [final_seq],
            "design_dist": [final_distance]
        })
        if len(self.summary_df) == 0:
            self.summary_df = next_row
        else:     
            self.summary_df = concat((self.summary_df, next_row))
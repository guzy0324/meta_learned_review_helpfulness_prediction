from functools import partialmethod
from types import TracebackType
from typing import Optional
from os import devnull
import sys

from tqdm import tqdm

# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(devnull, 'w')
        self._original_stderr = sys.stderr
        sys.stderr = open(devnull, 'w')
        self._original_tqdm_init = tqdm.__init__
        # https://stackoverflow.com/questions/37091673/silence-tqdms-output-while-running-tests-or-running-the-code-via-cron
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    def __exit__(self, exc_type: Optional[BaseException], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr.close()
        sys.stderr = self._original_stderr
        tqdm.__init__ = self._original_tqdm_init

if __name__ == "__main__":
    from tqdm import tqdm

    with HiddenPrints():
        print("This will not be printed")

    print("This will be printed as before")

    with HiddenPrints():
        for _ in tqdm(range(10)):
            pass

    for _ in tqdm(range(10)):
        pass

"""A small synthetic trace bank: cell bs2, all 28 splits, two schedulers, 20 windows.

eMBB throughput grows with its PRBs (as in ColO-RAN); proportional fair is 10 %
faster than round robin so a scheduler switch is visible.
"""

import pandas as pd
import pytest

from ran_emulator.configs import SPLITS
from ran_emulator.scenarios import CellSpec, Scenario
from ran_emulator.traces import TraceBank

WINDOWS = 20


def _rows():
    for sched_id, sched, gain in ((0, "round_robin", 1.0), (2, "proportional_fair", 1.1)):
        for tr, split in enumerate(SPLITS):
            run = f"sched{sched_id}/tr{tr}/exp1"
            for w in range(WINDOWS):
                for slice_id, rate, util, delay in (
                    ("eMBB", 0.1 * split.embb * gain + 0.001 * w, 0.9, 600 - 15 * split.embb),
                    ("MTC", 0.06, 0.2, 2.0),
                    ("URLLC", 0.12, 0.1, 0.5),
                ):
                    yield {
                        "record_id": f"{run.replace('/', '-')}-bs2-t{50 + w:06.1f}-{slice_id}",
                        "cell_id": "bs2",
                        "slice_id": slice_id,
                        "run": run,
                        "window_start_s": 50.0 + w,
                        "window_s": 1.0,
                        "slice_prb": split.of(slice_id),
                        "cell_prb": 50,
                        "scheduler": sched,
                        "active_ues": 2,
                        "dl_throughput_mbps": rate,
                        "dl_throughput_per_ue_mbps": rate / 2,
                        "dl_buffer_bytes": 1000.0,
                        "queueing_delay_ms": delay,
                        "prb_utilisation": util,
                        "cell_spare_prb": 5.0,
                        "dl_cqi": 9.0,
                        "dl_mcs": 15.0,
                        "ul_sinr_db": None if w % 5 == 0 else 15.0,
                    }


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return pd.DataFrame(list(_rows()))


@pytest.fixture(scope="session")
def bank(frame) -> TraceBank:
    return TraceBank.from_frame(frame)


@pytest.fixture
def make_scenario():
    def _make(tr=9, scheduler="round_robin", events=(), policies=None, start_offset=0, seed=0):
        return Scenario(
            name="t",
            seed=seed,
            cells={
                "bs2": CellSpec(
                    scheduler=scheduler, tr=tr, start_offset=start_offset, policies=policies or {}
                )
            },
            events=list(events),
        )

    return _make

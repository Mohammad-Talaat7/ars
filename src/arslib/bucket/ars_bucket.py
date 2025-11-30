"""Float bucket-based ARS implementation."""

import bisect
import math
from collections.abc import Iterable
from typing import override

from arslib.base.base_sorter import BaseSorter
from arslib.base.run import Run
from arslib.utils.adjacency_float import runs_are_adjacent_float
from arslib.utils.logger import setup_logger
from arslib.utils.merge_decision_float import merge_decision_float

logger = setup_logger("ARSBucket", "ars_bucket.log")

# Assumes logger, Run, BaseSorter, merge_decision_float, is_adjacent_left_float, is_adjacent_right_float are available


class ARSBucket(BaseSorter[float]):
    """Bucket-based ARS sorter for floats.

    Performance mode: minimal overhead in hot paths, but includes important logs
    and required safety checks (NaN detection, run/bucket updates).
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        super().__init__()
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        self.tol: float = float(tolerance)
        self.bucket_width: float = self.tol
        self.buckets: dict[int, list[int]] = {}  # bucket_index -> list of run_ids
        self.run_map: dict[int, Run[float]] = {}  # run_id -> Run

        # run_id -> (b_start, b_end)
        self.run_bucket_span: dict[int, tuple[int, int]] = {}
        self.next_run_id: int = 0
        self._current_index: int = (
            0  # index of value currently being processed (for NaN reporting)
        )

        logger.debug(
            f"ARSBucket init: tol={self.tol}, bucket_width={self.bucket_width}"
        )

    # -------------------------
    # Utility: bucket index
    # -------------------------
    def _bucket_index(self, value: float) -> int:
        # Use math.floor to handle negatives correctly
        return math.floor(value / self.bucket_width)

    # -------------------------
    # Bucket management
    # -------------------------
    def _add_run_to_buckets(self, run_id: int, run: Run[float]) -> None:
        # compute span and add run_id to each bucket in span
        b_start = self._bucket_index(run.start)
        b_end = self._bucket_index(run.end)
        self.run_bucket_span[run_id] = (b_start, b_end)
        for b in range(b_start, b_end + 1):
            lst = self.buckets.get(b)
            if lst is None:
                self.buckets[b] = [run_id]
            else:
                lst.append(run_id)

    def _remove_run_from_buckets(self, run_id: int) -> None:
        span = self.run_bucket_span.get(run_id)
        if not span:
            return
        b_start, b_end = span
        for b in range(b_start, b_end + 1):
            lst = self.buckets.get(b)
            if not lst:
                continue
            # remove run_id in-place, compact the list
            # use swap-pop removal for speed.
            try:
                idx = lst.index(run_id)
            except ValueError:
                continue
            # swap-remove
            last = lst[-1]
            lst[idx] = last
            _ = lst.pop()
            if not lst:
                del self.buckets[b]
        # drop stored span
        _ = self.run_bucket_span.pop(run_id, None)

    def _update_run_buckets_if_needed(self, run_id: int, run: Run[float]) -> None:
        old_span = self.run_bucket_span.get(run_id)
        new_start = self._bucket_index(run.start)
        new_end = self._bucket_index(run.end)
        new_span = (new_start, new_end)
        if old_span == new_span:
            return
        # remove old and add new span
        if old_span is not None:
            self._remove_run_from_buckets(run_id)
        self._add_run_to_buckets(run_id, run)

    # -------------------------
    # Run creation / insertion
    # -------------------------
    def _create_run_with_id(self, value: float) -> int:
        run = self._create_run(value)
        run_id = self.next_run_id
        self.next_run_id += 1
        self.run_map[run_id] = run
        # Add to buckets
        self._add_run_to_buckets(run_id, run)
        logger.debug(f"_create_run_with_id id={run_id} run={run}")
        return run_id

    def _insert_into_run(self, run: Run[float], value: float) -> None:
        # Fast paths: near boundaries
        # Use local bindings for speed
        start = run.start
        end = run.end
        if value >= end:
            run.append_right(value)
            return
        if value <= start:
            run.append_left(value)
            return
        # Otherwise insert in correct sorted position (slower)
        arr = run.to_list()
        idx = bisect.bisect_left(arr, value)
        run.insert_at(idx, value)

    # -------------------------
    # Core processing
    # -------------------------
    @override
    def sort(self, data: Iterable[float]) -> list[float]:
        """Override sort to track indices for NaN reporting and performance."""
        lst = list(data)
        logger.debug(
            f"ARSBucketFloatSorter.sort called with {len(lst)} items; tol={self.tol}"
        )
        self.on_start(lst)

        # reset internal structures (allow reuse of sorter object)
        self.buckets.clear()
        self.run_map.clear()
        self.run_bucket_span.clear()
        self.next_run_id = 0
        self._current_index = 0
        self.runs: list[Run[float]] = []

        for i, value in enumerate(lst):
            self._current_index = i
            self.on_value_insert(value)
            self._process_value(value)

        result = self._get_output()
        self.on_finish(result)
        logger.debug(f"ARSBucketFloatSorter.sort finished output_size={len(result)}")
        return result

    def _merge_adjacent_runs(self, base_run_id: int) -> int:
        """Merge any runs adjacent to run `base_run_id` (transitively).

        Using linear sorted-merge + rebuilding a new Run. Returns the base_run_id
        (stays the same key) after merging all neighbors.
        """
        if base_run_id not in self.run_map:
            return base_run_id

        changed = True
        # We'll loop until no neighboring run remains adjacent (transitive closure)
        while changed:
            changed = False
            base_run = self.run_map.get(base_run_id)
            if base_run is None:
                break

            # compute bucket span for the base run
            b_start = self._bucket_index(base_run.start)
            b_end = self._bucket_index(base_run.end)

            # Collect candidate run ids that overlap the base run span
            cand_ids: set[int] = set()
            # also include +/- 1 bucket to be safe for boundary cases
            for nb in range(b_start - 1, b_end + 2):
                lst = self.buckets.get(nb)
                if lst is not None:
                    cand_ids.update(lst)

            # Remove base id if present
            cand_ids.discard(base_run_id)

            # For deterministic behavior, iterate sorted by run.start
            cand_list = sorted(
                (cid for cid in cand_ids if cid in self.run_map),
                key=lambda cid: self.run_map[cid].start,
            )

            for cid in cand_list:
                other = self.run_map.get(cid)
                if other is None:
                    continue
                # If they are adjacent (within tolerance), merge other into base_run
                if runs_are_adjacent_float(base_run, other, self.tol):
                    # Prepare merged list in linear time
                    left_vals = base_run.to_list()
                    right_vals = other.to_list()

                    # Merge two sorted lists (linear)
                    i = j = 0
                    merged: list[float] = []
                    len_l, len_r = len(left_vals), len(right_vals)
                    while i < len_l and j < len_r:
                        if left_vals[i] <= right_vals[j]:
                            merged.append(left_vals[i])
                            i += 1
                        else:
                            merged.append(right_vals[j])
                            j += 1
                    if i < len_l:
                        merged.extend(left_vals[i:])
                    else:
                        merged.extend(right_vals[j:])

                    # Rebuild Run for the merged values
                    new_run = Run(merged)

                    # Remove old bucket spans for both runs
                    self._remove_run_from_buckets(base_run_id)
                    self._remove_run_from_buckets(cid)

                    # Replace base run in run_map with new_run
                    self.run_map[base_run_id] = new_run

                    # Remove the merged run id (cid)
                    if cid in self.run_map:
                        del self.run_map[cid]
                    # remove its stored span (safety)
                    _ = self.run_bucket_span.pop(cid, None)

                    # Add new span for base_run
                    self._add_run_to_buckets(base_run_id, new_run)

                    # Update base_run reference and flag that something changed
                    base_run = new_run
                    changed = True

                    # Update the "runs" view for compatibility
                    self.runs = list(self.run_map.values())

                    # Call hook with original other (we may keep a ref)
                    self.on_run_merge(base_run, other, base_run)

                    # After merge, break to recompute candidate set (safe)
                    break

            # loop repeats until no merges happen
        return base_run_id

    @override
    def _process_value(self, value: float) -> None:
        """Insert value into bucket-run structure, possibly create/merge runs."""
        i = self._current_index

        # NaN check with index in error message
        if math.isnan(value):
            logger.error(f"NaN encountered at input index {i}")
            raise ValueError(f"NaN found at index {i}")

        # Fast path: no runs yet
        if not self.run_map:
            _ = self._create_run_with_id(value)
            # keep runs list for compatibility; we'll maintain it lazily
            self.runs = list(self.run_map.values())
            return

        # Determine candidate run ids using tolerance window
        tol = self.tol
        lo_bucket = self._bucket_index(value - tol)
        hi_bucket = self._bucket_index(value + tol)

        candidates: list[int] = []
        for nb in range(lo_bucket, hi_bucket + 1):
            lst = self.buckets.get(nb)
            if lst is not None:
                candidates.extend(lst)

        if not candidates:
            # no nearby runs -> create new run
            _ = self._create_run_with_id(value)
            self.runs = list(self.run_map.values())
            return

        # deduplicate candidate ids (small lists so set conversion is cheap)
        cand_ids: set[int] = set(candidates)
        cand_list: list[int] = sorted(
            (rid for rid in cand_ids if rid in self.run_map),
            key=lambda rid: self.run_map[rid].start,
        )

        # Find best left_run (max end <= value) and right_run (min start >= value)
        left_id: int | None = None
        right_id: int | None = None
        left_end_max = -math.inf
        right_start_min = math.inf
        inside_id: int | None = None

        tol = self.tol  # local copy

        for rid in cand_list:
            run = self.run_map.get(rid)
            if run is None:
                continue
            rs = run.start
            re = run.end
            # value inside run (including tolerance)
            if (rs - tol) <= value <= (re + tol):
                inside_id = rid
                break  # definitive hit, prefer inside
            # run to the left of value
            if re < value:
                if re > left_end_max:
                    left_end_max = re
                    left_id = rid
            # run to the right of value
            if rs > value:
                if rs < right_start_min:
                    right_start_min = rs
                    right_id = rid

        # If value falls inside an existing run (including tolerance), insert in that run
        if inside_id is not None:
            run = self.run_map[inside_id]
            # insertion (fast path handles boundaries)
            self._insert_into_run(run, value)
            # update buckets if bounds changed
            self._update_run_buckets_if_needed(inside_id, run)
            _ = self._merge_adjacent_runs(inside_id)
            return

        # Map run_id -> actual Run or None for left/right
        left_run = self.run_map[left_id] if left_id is not None else None
        right_run = self.run_map[right_id] if right_id is not None else None

        # Decide merge direction using provided helper
        decision = merge_decision_float(value, left_run, right_run, tol)

        if decision == "none":
            # create a new run
            _ = self._create_run_with_id(value)
            self.runs = list(self.run_map.values())
            return

        if decision == "left":
            # insert into left run (right side of left)
            assert left_id is not None
            assert left_run is not None
            self._insert_into_run(left_run, value)
            # update buckets if needed
            self._update_run_buckets_if_needed(left_id, left_run)
            _ = self._merge_adjacent_runs(left_id)
            return

        if decision == "right":
            assert right_id is not None
            assert right_run is not None
            self._insert_into_run(right_run, value)
            self._update_run_buckets_if_needed(right_id, right_run)
            _ = self._merge_adjacent_runs(right_id)
            return

        # decision == "both"
        # We must merge left and right runs with the value in between them.
        # Prefer merging into left_run by appending right into left then insert value appropriately.
        assert left_run is not None and right_run is not None
        assert left_id is not None and right_id is not None
        # Merge right into left (left becomes the merged run)
        # Remove right_id from buckets and run_map, then merge blocks
        # We need run IDs for both
        lid = left_id
        rid = right_id

        # Save before deletion
        right = self.run_map[rid]

        # remove right run from buckets before merging to avoid stale references
        self._remove_run_from_buckets(rid)

        # perform in-place merge: left.merge_right_run(right)
        left_run.merge_right_run(right)
        # update left run with inserted value if necessary
        self._insert_into_run(left_run, value)
        # ------------------------------------------------------------------------

        # delete right run from run_map and span
        del self.run_map[rid]
        _ = self.run_bucket_span.pop(rid, None)

        # update buckets for left run (new boundaries)
        self._update_run_buckets_if_needed(lid, left_run)

        # notify via hook
        self.on_run_merge(left_run, right, left_run)
        # keep runs list updated
        self.runs = list(self.run_map.values())
        logger.debug(f"merged runs {lid} and {rid} with inserted value {value}")

    # -------------------------
    # Output assembly
    # -------------------------
    @override
    def _get_output(self) -> list[float]:
        """Return flattened sorted list by ordering runs by start."""
        if not self.run_map:
            return []
        # sort runs by start (start is cached in Run)
        runs_sorted = sorted(self.run_map.values(), key=lambda r: r.start)
        out: list[float] = []
        for r in runs_sorted:
            out.extend(r.to_list())
        return out

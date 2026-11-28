"""M4.1 reference model and parameter contract for the ORCUS inference tile.

This module is a pure-Python specification. It deliberately contains no RTL,
memory implementation, or synthesis assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from .lane_scaling import accumulator_width as row_dot_width

SUPPORTED_LANES = (4, 8, 16, 32)
INT4_MIN = -8
INT4_MAX = 7
MEMORY_BITS_PER_OPERAND = 16_384

# Execution-verified Run #267 Dadda core latencies, indexed by lane count.
DADDA_CORE_LATENCY_CYCLES = {4: 6, 8: 8, 16: 10, 32: 11}


def ceil_log2(value: int) -> int:
    """Return ceil(log2(value)) exactly for positive integer value."""
    if type(value) is not int or value < 1:
        raise ValueError(f"value must be a positive integer, got {value!r}")
    return (value - 1).bit_length()


def signed_bounds(width: int) -> tuple[int, int]:
    if type(width) is not int or width < 1:
        raise ValueError(f"signed width must be a positive integer, got {width!r}")
    return (-(1 << (width - 1)), (1 << (width - 1)) - 1)


@dataclass(frozen=True)
class TileConfig:
    """Compile-time M4.1 configuration for one supported lane count."""

    lanes: int = 32
    memory_bits_per_operand: int = MEMORY_BITS_PER_OPERAND

    def __post_init__(self) -> None:
        if type(self.lanes) is not int or self.lanes not in SUPPORTED_LANES:
            raise ValueError(f"lanes must be one of {SUPPORTED_LANES}, got {self.lanes!r}")
        if type(self.memory_bits_per_operand) is not int:
            raise ValueError("memory_bits_per_operand must be an integer")
        if self.memory_bits_per_operand != MEMORY_BITS_PER_OPERAND:
            raise ValueError(
                f"v1 requires exactly {MEMORY_BITS_PER_OPERAND} logical bits per operand"
            )
        if self.memory_bits_per_operand % self.word_width_bits:
            raise ValueError("operand memory capacity must be an integer number of words")

    @property
    def word_width_bits(self) -> int:
        return self.lanes * 4

    @property
    def memory_depth_words(self) -> int:
        return self.memory_bits_per_operand // self.word_width_bits

    @property
    def maximum_k(self) -> int:
        return self.memory_depth_words

    @property
    def dot_width_bits(self) -> int:
        # Existing ORCUS lane-scaling helper computes 8 + ceil(log2(lanes)).
        return row_dot_width(self.lanes)

    @property
    def accumulator_width_bits(self) -> int:
        return self.dot_width_bits + ceil_log2(self.memory_depth_words)

    @property
    def address_width_bits(self) -> int:
        return ceil_log2(self.memory_depth_words)

    @property
    def k_length_width_bits(self) -> int:
        # Must represent every value from zero through the full memory depth.
        return ceil_log2(self.memory_depth_words + 1)

    @property
    def core_latency_cycles(self) -> int:
        return DADDA_CORE_LATENCY_CYCLES[self.lanes]

    @property
    def total_memory_bits(self) -> int:
        return 2 * self.memory_bits_per_operand

    @property
    def int4_values_per_operand(self) -> int:
        return self.memory_bits_per_operand // 4

    def validate_request(self, base_addr: int, k_length: int) -> tuple[int, ...]:
        """Validate a non-wrapping contiguous request and return its addresses."""
        if type(base_addr) is not int:
            raise ValueError("base_addr must be an integer")
        if type(k_length) is not int:
            raise ValueError("k_length must be an integer")
        if not 0 <= base_addr < self.memory_depth_words:
            raise ValueError(
                f"base_addr must be in [0, {self.memory_depth_words - 1}]"
            )
        if not 1 <= k_length <= self.maximum_k:
            raise ValueError(f"k_length must be in [1, {self.maximum_k}]")
        end_exclusive = base_addr + k_length
        if end_exclusive > self.memory_depth_words:
            raise ValueError("request crosses the end of scratchpad; address wrap is forbidden")
        return tuple(range(base_addr, end_exclusive))

    def request_to_result_latency_cycles(self, k_length: int) -> int:
        """Derived target from start acceptance edge to result-valid edge.

        The first synchronous read is issued at the acceptance edge. K rows are
        issued one per cycle; the last row enters the L-cycle Dadda core after
        the one-cycle read; one output register follows the core.
        """
        self.validate_request(0, k_length)
        return k_length + self.core_latency_cycles + 1


def dot_product_bounds(lanes: int) -> tuple[int, int]:
    TileConfig(lanes)
    # One signed INT4 product is in [-56, 64].
    return (-56 * lanes, 64 * lanes)


def mac_accumulator_bounds(lanes: int, k_length: int) -> tuple[int, int]:
    config = TileConfig(lanes)
    config.validate_request(0, k_length)
    return (-56 * lanes * k_length, 64 * lanes * k_length)


def _int4_row(values: Iterable[int], lanes: int, label: str) -> tuple[int, ...]:
    try:
        row = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable of signed INT4 values") from exc
    if len(row) != lanes:
        raise ValueError(f"{label} must contain exactly {lanes} values, got {len(row)}")
    for index, value in enumerate(row):
        if type(value) is not int or not INT4_MIN <= value <= INT4_MAX:
            raise ValueError(
                f"{label}[{index}] must be a signed INT4 integer in "
                f"[{INT4_MIN}, {INT4_MAX}], got {value!r}"
            )
    return row


def signed_int4_dot_product(
    a_values: Iterable[int], b_values: Iterable[int], lanes: int = 32
) -> int:
    """Exact signed dot product of one row of packed INT4 operands."""
    config = TileConfig(lanes)
    a_row = _int4_row(a_values, config.lanes, "a_values")
    b_row = _int4_row(b_values, config.lanes, "b_values")
    result = sum(a * b for a, b in zip(a_row, b_row))
    lower, upper = signed_bounds(config.dot_width_bits)
    if not lower <= result <= upper:
        raise AssertionError("dot product exceeded the configured signed result width")
    return result


def signed_int4_mac_accumulate(
    a_rows: Iterable[Iterable[int]],
    b_rows: Iterable[Iterable[int]],
    lanes: int = 32,
) -> int:
    """Exact full-precision sum of K consecutive signed INT4 row dot products."""
    config = TileConfig(lanes)
    try:
        a_batch = tuple(a_rows)
        b_batch = tuple(b_rows)
    except TypeError as exc:
        raise ValueError("a_rows and b_rows must be iterables of rows") from exc
    if len(a_batch) != len(b_batch):
        raise ValueError("a_rows and b_rows must contain the same number of rows")
    config.validate_request(0, len(a_batch))
    result = sum(
        signed_int4_dot_product(a_row, b_row, config.lanes)
        for a_row, b_row in zip(a_batch, b_batch)
    )
    lower, upper = signed_bounds(config.accumulator_width_bits)
    if not lower <= result <= upper:
        raise AssertionError("MAC result exceeded the configured signed accumulator width")
    return result


def pack_operand_word(values: Iterable[int], lanes: int = 32) -> int:
    """Pack lane 0 into bits [3:0], lane 1 into [7:4], and so on."""
    config = TileConfig(lanes)
    row = _int4_row(values, config.lanes, "values")
    packed = 0
    for lane, value in enumerate(row):
        packed |= (value & 0xF) << (4 * lane)
    return packed


def unpack_operand_word(word: int, lanes: int = 32) -> tuple[int, ...]:
    """Unpack a word using the v1 least-significant-nibble-first lane order."""
    config = TileConfig(lanes)
    if type(word) is not int or not 0 <= word < (1 << config.word_width_bits):
        raise ValueError(f"word must be an unsigned {config.word_width_bits}-bit integer")
    values = []
    for lane in range(config.lanes):
        nibble = (word >> (4 * lane)) & 0xF
        values.append(nibble - 16 if nibble & 0x8 else nibble)
    return tuple(values)

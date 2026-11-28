import unittest

from orcus.dadda_pipeline import generate_dadda_pipelined
from orcus.inference_tile import (
    DADDA_CORE_LATENCY_CYCLES,
    MEMORY_BITS_PER_OPERAND,
    SUPPORTED_LANES,
    TileConfig,
    dot_product_bounds,
    mac_accumulator_bounds,
    pack_operand_word,
    signed_bounds,
    signed_int4_dot_product,
    signed_int4_mac_accumulate,
    unpack_operand_word,
)


class InferenceTileConfigTests(unittest.TestCase):
    def test_parameter_matrix_and_exact_memory_dimensions(self):
        expected = {
            4: (16, 1024, 10, 20, 6, 10, 11),
            8: (32, 512, 11, 20, 8, 9, 10),
            16: (64, 256, 12, 20, 10, 8, 9),
            32: (128, 128, 13, 20, 11, 7, 8),
        }
        for lanes in SUPPORTED_LANES:
            with self.subTest(lanes=lanes):
                config = TileConfig(lanes)
                word, depth, dot_width, acc_width, latency, addr_width, k_width = expected[lanes]
                self.assertEqual(config.word_width_bits, word)
                self.assertEqual(config.memory_depth_words, depth)
                self.assertEqual(config.dot_width_bits, dot_width)
                self.assertEqual(config.accumulator_width_bits, acc_width)
                self.assertEqual(config.core_latency_cycles, latency)
                self.assertEqual(config.address_width_bits, addr_width)
                self.assertEqual(config.k_length_width_bits, k_width)
                self.assertEqual(config.total_memory_bits, 32768)
                self.assertEqual(config.int4_values_per_operand, 4096)
                self.assertEqual(config.memory_bits_per_operand, MEMORY_BITS_PER_OPERAND)

    def test_32_lane_stress_configuration_and_latency_formula(self):
        config = TileConfig(32)
        self.assertEqual(config.word_width_bits, 128)
        self.assertEqual(config.memory_depth_words, 128)
        self.assertEqual(config.dot_width_bits, 13)
        self.assertEqual(config.accumulator_width_bits, 20)
        self.assertEqual(config.core_latency_cycles, 11)
        self.assertEqual(config.request_to_result_latency_cycles(1), 13)
        self.assertEqual(config.request_to_result_latency_cycles(2), 14)
        self.assertEqual(config.request_to_result_latency_cycles(128), 140)
        # L+2 is the K=1 special case, not the general K-row transaction.
        self.assertEqual(config.request_to_result_latency_cycles(1), config.core_latency_cycles + 2)

    def test_core_latency_constants_match_the_validated_dadda_generator(self):
        for lanes, expected_latency in DADDA_CORE_LATENCY_CYCLES.items():
            with self.subTest(lanes=lanes):
                self.assertEqual(generate_dadda_pipelined(lanes).latency, expected_latency)
                self.assertEqual(TileConfig(lanes).core_latency_cycles, expected_latency)

    def test_only_supported_lane_counts_and_fixed_capacity_are_accepted(self):
        for lanes in (0, 1, 2, 3, 5, 12, 64, True):
            with self.subTest(lanes=lanes):
                with self.assertRaises(ValueError):
                    TileConfig(lanes)
        with self.assertRaises(ValueError):
            TileConfig(32, memory_bits_per_operand=8192)
        with self.assertRaises(ValueError):
            TileConfig(32, memory_bits_per_operand=32768)


class InferenceTileArithmeticTests(unittest.TestCase):
    def test_32_lane_dot_product_signed_extrema(self):
        lanes = 32
        self.assertEqual(dot_product_bounds(lanes), (-1792, 2048))
        maximum = signed_int4_dot_product([-8] * lanes, [-8] * lanes, lanes)
        minimum = signed_int4_dot_product([-8] * lanes, [7] * lanes, lanes)
        self.assertEqual(maximum, 2048)
        self.assertEqual(minimum, -1792)
        lower, upper = signed_bounds(TileConfig(lanes).dot_width_bits)
        self.assertLessEqual(lower, minimum)
        self.assertGreaterEqual(upper, maximum)

    def test_full_depth_32_lane_mac_signed_extrema_fit_20_bits(self):
        config = TileConfig(32)
        a_max = [-8] * config.lanes
        b_max = [-8] * config.lanes
        a_min = [-8] * config.lanes
        b_min = [7] * config.lanes
        maximum = signed_int4_mac_accumulate(
            [a_max] * config.maximum_k, [b_max] * config.maximum_k, 32
        )
        minimum = signed_int4_mac_accumulate(
            [a_min] * config.maximum_k, [b_min] * config.maximum_k, 32
        )
        self.assertEqual(mac_accumulator_bounds(32, config.maximum_k), (-229376, 262144))
        self.assertEqual(minimum, -229376)
        self.assertEqual(maximum, 262144)
        lower, upper = signed_bounds(config.accumulator_width_bits)
        self.assertGreaterEqual(minimum, lower)
        self.assertLessEqual(maximum, upper)

    def test_accumulator_bounds_fit_for_every_supported_configuration(self):
        for lanes in SUPPORTED_LANES:
            config = TileConfig(lanes)
            lower, upper = mac_accumulator_bounds(lanes, config.maximum_k)
            signed_lower, signed_upper = signed_bounds(config.accumulator_width_bits)
            with self.subTest(lanes=lanes):
                self.assertGreaterEqual(lower, signed_lower)
                self.assertLessEqual(upper, signed_upper)
                self.assertEqual(config.accumulator_width_bits, 20)

    def test_mac_model_matches_sum_of_individual_row_dot_products(self):
        a_rows = [[-8, 7, -1, 3] * 8, [2, -3, 4, -5] * 8]
        b_rows = [[7, -8, 2, -4] * 8, [-6, 5, -7, 1] * 8]
        expected = sum(
            signed_int4_dot_product(a, b, 32)
            for a, b in zip(a_rows, b_rows)
        )
        self.assertEqual(signed_int4_mac_accumulate(a_rows, b_rows, 32), expected)

    def test_rejects_mismatched_rows_and_invalid_int4_values(self):
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            signed_int4_mac_accumulate([[0] * 32], [], 32)
        with self.assertRaisesRegex(ValueError, "exactly 32 values"):
            signed_int4_dot_product([0] * 31, [0] * 32, 32)
        for bad in (-9, 8, True, 1.5):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "signed INT4"):
                    signed_int4_dot_product([bad] + [0] * 31, [0] * 32, 32)

    def test_rejects_zero_or_over_depth_k(self):
        config = TileConfig(32)
        row = [0] * 32
        with self.assertRaisesRegex(ValueError, "k_length"):
            signed_int4_mac_accumulate([], [], 32)
        with self.assertRaisesRegex(ValueError, "k_length"):
            signed_int4_mac_accumulate([row] * 129, [row] * 129, 32)


class InferenceTilePackingAndAddressTests(unittest.TestCase):
    def test_32_lane_pack_unpack_and_nibble_order(self):
        values = [((lane % 16) - 8) for lane in range(32)]
        packed = pack_operand_word(values, 32)
        self.assertEqual(packed & 0xF, values[0] & 0xF)
        self.assertEqual((packed >> 4) & 0xF, values[1] & 0xF)
        self.assertEqual(unpack_operand_word(packed, 32), tuple(values))

    def test_pack_unpack_round_trip_for_every_supported_width(self):
        for lanes in SUPPORTED_LANES:
            values = [((lane * 7) % 16) - 8 for lane in range(lanes)]
            with self.subTest(lanes=lanes):
                self.assertEqual(unpack_operand_word(pack_operand_word(values, lanes), lanes),
                                 tuple(values))

    def test_rejects_out_of_range_packed_words_and_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "unsigned 128-bit"):
            unpack_operand_word(1 << 128, 32)
        with self.assertRaisesRegex(ValueError, "signed INT4"):
            pack_operand_word([0] * 31 + [8], 32)

    def test_non_wrapping_address_windows_and_boundary_cases(self):
        config = TileConfig(32)
        self.assertEqual(config.validate_request(0, 1), (0,))
        self.assertEqual(config.validate_request(127, 1), (127,))
        self.assertEqual(config.validate_request(0, 128), tuple(range(128)))
        self.assertEqual(config.validate_request(32, 96), tuple(range(32, 128)))
        with self.assertRaisesRegex(ValueError, "base_addr"):
            config.validate_request(128, 1)
        with self.assertRaisesRegex(ValueError, "k_length"):
            config.validate_request(0, 0)
        with self.assertRaisesRegex(ValueError, "crosses the end"):
            config.validate_request(127, 2)


if __name__ == "__main__":
    unittest.main()

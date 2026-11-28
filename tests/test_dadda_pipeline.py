import unittest

from benchmarks.dadda_pipeline import (
    LANES, VECTOR_COUNT, build_control_miter, build_initial_heap_miter,
    build_stage_partition, build_stage_column_miter, build_final_cpa_miter,
    validate_stage_bit_partition, build_sequential_miter,
    build_streaming_testbench, _vectors, build_atomic_compressor_miter,
    validate_stage_netlist_wiring,
)
from orcus.dadda_pipeline import generate_dadda_pipelined
from orcus.lane_scaling import accumulator_width


class DaddaPipelineGeneratorTests(unittest.TestCase):
    def test_all_lane_configurations_have_complete_stage_boundaries(self):
        for lanes in LANES:
            with self.subTest(lanes=lanes):
                design = generate_dadda_pipelined(lanes)
                self.assertEqual(design.width, accumulator_width(lanes))
                self.assertEqual(design.latency, len(design.stage_targets))
                self.assertEqual(len(design.manifest["stage_boundaries"]), design.latency)
                self.assertGreater(design.latency, 0)
                for stage, boundary in enumerate(design.manifest["stage_boundaries"]):
                    self.assertEqual(boundary["stage"], stage)
                    self.assertEqual(
                        boundary["registered_bit_count"],
                        sum(len(column) for column in boundary["registered_heap"]),
                    )
                    expected = [
                        [f"st{stage}_q_{token}" for token in column]
                        for column in boundary["output_heap"]
                    ]
                    self.assertEqual(boundary["registered_heap"], expected)

    def test_pipeline_interface_reset_and_latency_metadata(self):
        for lanes in LANES:
            design = generate_dadda_pipelined(lanes)
            self.assertIn(f"module orcus_dadda_pipelined_{lanes}(", design.source)
            self.assertIn("input wire reset", design.source)
            self.assertIn("input wire in_valid", design.source)
            self.assertIn("output reg out_valid", design.source)
            self.assertIn("if (reset)", design.source)
            self.assertIn(f"localparam integer LATENCY = {design.latency};", design.source)
            self.assertIn(f"out_valid <= valid_pipe[{design.latency - 1}];", design.source)
            self.assertIn("one input transaction per cycle", design.manifest["throughput"])

    def test_initial_baugh_wooley_heap_matches_signed_mac(self):
        for lanes in LANES:
            for lane_index in range(lanes):
                with self.subTest(lanes=lanes, lane=lane_index):
                    source = build_initial_heap_miter(lanes, lane_index)
                    self.assertIn(
                        f"module orcus_dadda_initial_heap_miter_{lanes}_{lane_index}",
                        source,
                    )
                    self.assertIn("initial_heap_sum", source)
                    self.assertIn("extended_product", source)
                    self.assertIn("wire signed [7:0] signed_a", source)
                    self.assertIn("wire signed [7:0] signed_b", source)
                    self.assertIn("signed_product = signed_a * signed_b", source)
                    self.assertIn("assign bad = (initial_heap_sum != extended_product)", source)
                    self.assertIn(f"bias_{lane_index}_", source)


    def test_each_stage_has_per_column_conservation_miters_and_exact_accounting(self):
        for lanes in LANES:
            design=generate_dadda_pipelined(lanes)
            for stage in range(design.latency):
                with self.subTest(lanes=lanes,stage=stage):
                    part=build_stage_partition(lanes,stage)
                    boundary=design.manifest["stage_boundaries"][stage]
                    self.assertEqual(part["input_bit_count"],sum(map(len,boundary["input_heap"])))
                    self.assertEqual(part["output_bit_count"],sum(map(len,boundary["output_heap"])))
                    self.assertEqual(part["consumed_input_bit_count"]+
                                     part["primary_pass_through_bit_count"],part["input_bit_count"])
                    self.assertEqual(part["output_bit_count"],part["pass_through_bit_count"]+
                        sum(1 for r in design.manifest["compressors"] if r["stage"]==stage))
                    for col in part["columns"]:
                        if not col["compressors"]:
                            continue
                        source=build_stage_column_miter(lanes,stage,col["column"])
                        self.assertIn(f"orcus_dadda_stage_column_miter_{lanes}_{stage}_{col['column']}",source)
                        self.assertIn("assign bad =",source)
                        if col["column"]==design.width-1:
                            self.assertIn("!= ",source)
                        else:
                            self.assertIn("conservation_inputs",source)
                            self.assertIn("conservation_outputs",source)

    def test_partition_accounting_rejects_dropped_duplicated_and_misweighted_bits(self):
        design=generate_dadda_pipelined(4)
        boundary=design.manifest["stage_boundaries"][0]
        compressors=[dict(r) for r in design.manifest["compressors"] if r["stage"]==0]
        inputs=[list(c) for c in boundary["input_heap"]]
        outputs=[list(c) for c in boundary["output_heap"]]
        width=design.width
        dropped=[list(c) for c in outputs]
        idx=next(i for i,c in enumerate(dropped) if c)
        dropped[idx].pop()
        with self.assertRaisesRegex(ValueError,"output partition mismatch"):
            validate_stage_bit_partition(inputs,dropped,compressors,width)
        duplicated=[list(c) for c in outputs]
        duplicated[idx].append(duplicated[idx][0])
        with self.assertRaisesRegex(ValueError,"appears more than once|duplicate output"):
            validate_stage_bit_partition(inputs,duplicated,compressors,width)
        misweighted=[dict(r) for r in compressors]
        carry=next(r for r in misweighted if r["carry"] is not None)
        carry["carry_column"]=carry["column"]
        with self.assertRaisesRegex(ValueError,"misweighted carry"):
            validate_stage_bit_partition(inputs,outputs,misweighted,width)
        duplicated_input=[dict(r) for r in compressors]
        duplicated_input[0]["inputs"]=list(duplicated_input[0]["inputs"])
        duplicated_input[0]["inputs"][1]=duplicated_input[0]["inputs"][0]
        with self.assertRaisesRegex(ValueError,"consumed more than once"):
            validate_stage_bit_partition(inputs,outputs,duplicated_input,width)

    def test_atomic_fa_ha_conservation_miters_are_exact_and_distinct(self):
        for kind, inputs in (("full_adder", ("a", "b", "cin")),
                             ("half_adder", ("a", "b"))):
            top, source = build_atomic_compressor_miter(kind)
            self.assertIn(f"module {top}(", source)
            self.assertIn("wire [2:0] lhs", source)
            self.assertIn("wire [2:0] rhs", source)
            self.assertIn("assign bad = (lhs != rhs)", source)
            for pin in inputs:
                self.assertIn(f"input wire {pin}", source)
        with self.assertRaisesRegex(ValueError, "unsupported compressor kind"):
            build_atomic_compressor_miter("unknown")

    def test_emitted_stage_netlist_matches_every_compressor_pin(self):
        design = generate_dadda_pipelined(32)
        for stage in range(design.latency):
            records = [r for r in design.manifest["compressors"] if r["stage"] == stage]
            audit = validate_stage_netlist_wiring(design.source, records, stage, design.manifest["stage_boundaries"][stage]["output_heap"])
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["compressors_checked"], len(records))

    def test_netlist_audit_rejects_miswired_fa_and_ha_pins(self):
        design = generate_dadda_pipelined(4)
        records = [r for r in design.manifest["compressors"] if r["stage"] == 0]
        fa = next(r for r in records if r["kind"] == "full_adder")
        x, y, z = fa["inputs"]
        bad_source = design.source.replace(
            f"wire {fa['sum']} = {x} ^ {y} ^ {z};",
            f"wire {fa['sum']} = {x} ^ {x} ^ {z};", 1)
        with self.assertRaisesRegex(ValueError, "sum pin/netlist mismatch"):
            validate_stage_netlist_wiring(bad_source, records, 0, design.manifest["stage_boundaries"][0]["output_heap"])
        ha = next(r for r in records if r["kind"] == "half_adder")
        x, y = ha["inputs"]
        bad_source = design.source.replace(
            f"wire {ha['carry']} = {x} & {y};",
            f"wire {ha['carry']} = {x} & {x};", 1)
        with self.assertRaisesRegex(ValueError, "carry pin/netlist mismatch"):
            validate_stage_netlist_wiring(bad_source, records, 0)

    def test_netlist_audit_rejects_dropped_or_duplicated_register_tokens(self):
        design = generate_dadda_pipelined(4)
        records = [r for r in design.manifest["compressors"] if r["stage"] == 0]
        output_heap = design.manifest["stage_boundaries"][0]["output_heap"]
        token = next(token for column in output_heap for token in column)
        qname = f"st0_q_{token}"
        register_line = f"      {qname} <= {token};"
        dropped = design.source.replace(register_line + "\n", "", 1)
        with self.assertRaisesRegex(ValueError, "register/pass-through wiring mismatch"):
            validate_stage_netlist_wiring(dropped, records, 0, output_heap)
        duplicated = design.source.replace(register_line,
            register_line + "\n" + register_line, 1)
        with self.assertRaisesRegex(ValueError, "register/pass-through wiring mismatch"):
            validate_stage_netlist_wiring(duplicated, records, 0, output_heap)

    def test_32_lane_stage0_column4_uses_atomic_composition_not_column_miter(self):
        design = generate_dadda_pipelined(32)
        records = [r for r in design.manifest["compressors"]
                   if r["stage"] == 0 and r["column"] == 4]
        self.assertTrue(records)
        self.assertTrue(all(r["kind"] in ("full_adder", "half_adder") for r in records))
        part = build_stage_partition(32, 0)
        column = part["columns"][4]
        self.assertEqual(column["compressors"], records)
        audit = validate_stage_netlist_wiring(design.source,
            [r for r in design.manifest["compressors"] if r["stage"] == 0], 0,
            design.manifest["stage_boundaries"][0]["output_heap"])
        self.assertEqual(audit["compressors_checked"],
                         sum(r["stage"] == 0 for r in design.manifest["compressors"]))
        top, fa = build_atomic_compressor_miter("full_adder")
        self.assertIn("orcus_atomic_fa_conservation", top)
        self.assertNotIn("orcus_dadda_stage_column_miter_32_0_4", fa)

    def test_final_cpa_is_proved_as_a_separate_miter_for_all_widths(self):
        for lanes in LANES:
            with self.subTest(lanes=lanes):
                source=build_final_cpa_miter(lanes)
                self.assertIn(f"orcus_dadda_final_cpa_miter_{lanes}",source)
                self.assertIn("final_cpa = final_row0 + final_row1",source)
                self.assertIn("final_cpa != weighted_output",source)

    def test_sequential_control_miter_checks_registered_output_boundary(self):
        for lanes in LANES:
            source, latency = build_control_miter(lanes)
            reference_depth = latency + 1
            self.assertIn(f"reg [{reference_depth - 1}:0] ref_valid_pipe;", source)
            self.assertIn(f"ref_valid_pipe[{reference_depth - 1}] <= ref_valid_pipe[{reference_depth - 2}];", source)
            self.assertIn(f"assign bad = (dut_valid != ref_valid_pipe[{reference_depth - 1}]);", source)
            self.assertIn(f"orcus_dadda_pipelined_{lanes} dut", source)
            self.assertIn("if (reset) ref_valid_pipe <= 0", source)
            self.assertNotIn("ref_data_pipe", source)
            self.assertEqual(build_sequential_miter(lanes), (source, latency))

    def test_dut_valid_depth_reference_depth_manifest_and_stream_latency_agree(self):
        for lanes in LANES:
            with self.subTest(lanes=lanes):
                design = generate_dadda_pipelined(lanes)
                latency = design.latency
                manifest_latency = design.manifest["latency_cycles"]
                control_source, control_latency = build_control_miter(lanes)
                stream_source, stream_latency = build_streaming_testbench(lanes)

                # L reduction-stage valid bits plus the separately registered
                # out_valid boundary form L+1 registered elements.  The input
                # sample reaches out_valid after L rising-edge intervals.
                self.assertEqual(manifest_latency, latency)
                self.assertEqual(control_latency, latency)
                self.assertEqual(stream_latency, latency)
                self.assertIn(f"reg [{latency - 1}:0] valid_pipe;", design.source)
                self.assertIn(f"out_valid <= valid_pipe[{latency - 1}];", design.source)
                dut_valid_depth = latency + int("output reg out_valid" in design.source)
                reference_depth = latency + 1
                self.assertEqual(dut_valid_depth, reference_depth)
                self.assertIn(f"reg [{latency}:0] ref_valid_pipe;", control_source)
                self.assertIn(f"ref_valid_pipe[{latency}]", control_source)
                # The builder unrolls the 512-vector schedule into literal
                # testbench statements; transaction zero must first be checked
                # at input index L, followed by the final L outputs in flush.
                self.assertIn(f"expected transaction 0", stream_source)
                self.assertIn("flush mismatch 0", stream_source)
                self.assertIn(f"flush mismatch {latency - 1}", stream_source)

    def test_streaming_testbench_has_512_back_to_back_vectors(self):
        for lanes in LANES:
            source, latency = build_streaming_testbench(lanes)
            self.assertEqual(VECTOR_COUNT, 512)
            self.assertIn("stream mismatch", source)
            self.assertIn("flush mismatch", source)
            self.assertIn("in_valid = 1'b1", source)
            self.assertIn("in_valid = 1'b0", source)
            self.assertIn("stale transaction leaked after reset", source)
            self.assertGreater(latency, 0)

    def test_reference_vectors_are_signed_and_width_bounded(self):
        for lanes in LANES:
            width = accumulator_width(lanes)
            lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
            vectors = _vectors(lanes)
            self.assertEqual(len(vectors), VECTOR_COUNT)
            self.assertTrue(all(lower <= expected <= upper for _, _, expected in vectors))


if __name__ == "__main__":
    unittest.main()

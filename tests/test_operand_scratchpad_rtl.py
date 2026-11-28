"""Cycle-accurate Icarus tests for ORCUS M4.2 A/B operand scratchpads."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from orcus.inference_tile import TileConfig, pack_operand_word

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "orcus" / "rtl" / "orcus_operand_scratchpad.sv"


def _sv_testbench(lanes: int) -> str:
    cfg = TileConfig(lanes)
    width = cfg.word_width_bits
    depth = cfg.memory_depth_words

    def packed(values: tuple[int, ...]) -> str:
        return f"{pack_operand_word(values, lanes):0{width // 4}x}"

    def row(seed: int, multiplier: int) -> tuple[int, ...]:
        return tuple(((seed + i * multiplier) % 16) - 8 for i in range(lanes))

    values = {
        "LANES": str(lanes),
        "WORD_WIDTH": str(width),
        "A0": packed(row(3, 7)),
        "B0": packed(row(1, 5)),
        "ALAST": packed(row(9, 11)),
        "BLAST": packed(row(4, 13)),
        "ALASTNEW": packed(row(2, 3)),
        "BMID": packed(row(6, 9)),
    }
    template = r"""`timescale 1ns/1ps
module tb;
    parameter integer LANES = @LANES@;
    localparam integer WORD_WIDTH = LANES * 4;
    localparam integer DEPTH = 16384 / WORD_WIDTH;
    localparam integer ADDR_WIDTH = $clog2(DEPTH);
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg [ADDR_WIDTH-1:0] wr_addr = 0;
    reg a_we = 0, b_we = 0, rd_en = 0;
    reg [WORD_WIDTH-1:0] a_wdata = 0, b_wdata = 0;
    reg [ADDR_WIDTH-1:0] rd_addr = 0;
    wire [WORD_WIDTH-1:0] a_rd_data, b_rd_data;

    orcus_operand_scratchpads #(.LANES(LANES)) dut (
        .clk(clk), .wr_addr(wr_addr), .a_we(a_we), .b_we(b_we),
        .a_wdata(a_wdata), .b_wdata(b_wdata), .rd_en(rd_en),
        .rd_addr(rd_addr), .a_rd_data(a_rd_data), .b_rd_data(b_rd_data)
    );

    initial begin
        if ($test$plusargs("COLLISION")) begin
            @(negedge clk);
            wr_addr = 0; rd_addr = 0; a_we = 1; rd_en = 1;
            a_wdata = {WORD_WIDTH{1'b1}};
            @(posedge clk); #1;
            $display("ERROR: collision guard was not triggered");
            $finish;
        end else begin
            // Initialize address zero in both operand memories.
            @(negedge clk);
            wr_addr = 0; a_we = 1; b_we = 1;
            a_wdata = @WORD_WIDTH@'h@A0@;
            b_wdata = @WORD_WIDTH@'h@B0@;
            @(posedge clk); #1;

            // Exercise the final legal address in both memories.
            @(negedge clk);
            wr_addr = DEPTH - 1;
            a_wdata = @WORD_WIDTH@'h@ALAST@;
            b_wdata = @WORD_WIDTH@'h@BLAST@;
            @(posedge clk); #1;

            // Concurrent compute read and host write at different addresses.
            // Only A writes; B must retain its final-address word.
            @(negedge clk);
            wr_addr = DEPTH - 1; rd_addr = 0; rd_en = 1;
            a_we = 1; b_we = 0; a_wdata = @WORD_WIDTH@'h@ALASTNEW@;
            @(posedge clk); #1;
            if (a_rd_data !== @WORD_WIDTH@'h@A0@)
                $fatal(1, "A synchronous read/packed alignment mismatch");
            if (b_rd_data !== @WORD_WIDTH@'h@B0@)
                $fatal(1, "B synchronous read/packed alignment mismatch");

            // Read back the boundary address after the A-only write.
            @(negedge clk);
            a_we = 0; b_we = 0; rd_addr = DEPTH - 1;
            @(posedge clk); #1;
            if (a_rd_data !== @WORD_WIDTH@'h@ALASTNEW@)
                $fatal(1, "A final-address write/read mismatch");
            if (b_rd_data !== @WORD_WIDTH@'h@BLAST@)
                $fatal(1, "B independent write-enable isolation mismatch");

            // Write B only at an interior address while reading A/B address 0.
            @(negedge clk);
            wr_addr = 1; rd_addr = 0; a_we = 0; b_we = 1;
            b_wdata = @WORD_WIDTH@'h@BMID@;
            @(posedge clk); #1;
            if (a_rd_data !== @WORD_WIDTH@'h@A0@)
                $fatal(1, "A changed during B-only write");
            if (b_rd_data !== @WORD_WIDTH@'h@B0@)
                $fatal(1, "B changed before interior-address read");

            @(negedge clk);
            b_we = 0; rd_addr = 1;
            @(posedge clk); #1;
            if (b_rd_data !== @WORD_WIDTH@'h@BMID@)
                $fatal(1, "B interior-address packed-word mismatch");

            // Read enable low must retain the registered output.
            @(negedge clk);
            rd_en = 0; rd_addr = 0;
            @(posedge clk); #1;
            if (b_rd_data !== @WORD_WIDTH@'h@BMID@)
                $fatal(1, "read data did not hold while rd_en was low");

            $display("PASS scratchpad LANES=%0d DEPTH=%0d WIDTH=%0d",
                     LANES, DEPTH, WORD_WIDTH);
            $finish;
        end
    end
endmodule
"""
    for key, value in values.items():
        template = template.replace("@" + key + "@", value)
    return template


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"),
                     "Icarus Verilog is required")
class OperandScratchpadRTLTests(unittest.TestCase):
    def _compile(self, output: Path, tb_path: Path) -> None:
        result = subprocess.run(
            ["iverilog", "-g2012", "-s", "tb", "-o", str(output),
             str(RTL), str(tb_path)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_supported_geometries_and_port_semantics(self) -> None:
        for lanes in (4, 8, 16, 32):
            with self.subTest(lanes=lanes), tempfile.TemporaryDirectory() as temp:
                temp_path = Path(temp)
                tb_path = temp_path / "tb.sv"
                sim_path = temp_path / "sim.out"
                tb_path.write_text(_sv_testbench(lanes))
                self._compile(sim_path, tb_path)
                result = subprocess.run(
                    ["vvp", str(sim_path)], cwd=ROOT, text=True,
                    capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"PASS scratchpad LANES={lanes}", result.stdout)

    def test_same_address_read_write_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            tb_path = temp_path / "tb.sv"
            sim_path = temp_path / "sim.out"
            tb_path.write_text(_sv_testbench(32))
            self._compile(sim_path, tb_path)
            result = subprocess.run(
                ["vvp", str(sim_path), "+COLLISION"], cwd=ROOT,
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("same-address read/write collision is prohibited",
                          result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

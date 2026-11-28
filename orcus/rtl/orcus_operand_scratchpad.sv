// ORCUS M4.2 — independent A/B operand scratchpads.
// One synchronous host-write port plus one synchronous compute-read port
// per operand (simple dual-port 1R1W RAM). Contents are intentionally not reset.
module orcus_operand_scratchpad #(
    parameter integer LANES = 32
) (
    input  wire clk,
    input  wire wr_en,
    input  wire [$clog2(16384/(LANES*4))-1:0] wr_addr,
    input  wire [LANES*4-1:0] wr_data,
    input  wire rd_en,
    input  wire [$clog2(16384/(LANES*4))-1:0] rd_addr,
    output reg  [LANES*4-1:0] rd_data
);
    localparam integer MEMORY_BITS = 16384;
    localparam integer WORD_WIDTH = LANES * 4;
    localparam integer DEPTH = MEMORY_BITS / WORD_WIDTH;

    (* ram_style = "block" *) reg [WORD_WIDTH-1:0] mem [0:DEPTH-1];

    // Synchronous registered read; independent-address read/write may overlap.
    // rd_data holds its previous value while rd_en is low.
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
        if (rd_en)
            rd_data <= mem[rd_addr];
    end

`ifndef SYNTHESIS
    initial begin
        if (!(LANES == 4 || LANES == 8 || LANES == 16 || LANES == 32))
            $fatal(1, "ORCUS scratchpad LANES must be 4, 8, 16, or 32");
        if (DEPTH * WORD_WIDTH != MEMORY_BITS)
            $fatal(1, "ORCUS scratchpad must contain exactly 16384 logical bits");
        if ((DEPTH & (DEPTH - 1)) != 0)
            $fatal(1, "ORCUS scratchpad depth must be a power of two");
    end

    // The v1 contract prohibits same-address collisions. Do not depend on
    // device-specific read-first/write-first behavior.
    always @(posedge clk) begin
        if (wr_en && rd_en && (wr_addr == rd_addr))
            $fatal(1, "ORCUS scratchpad same-address read/write collision is prohibited");
    end
`endif
endmodule


// Two independent operand memories. The host write address and compute read
// address are shared between A and B; write enables and packed data are not.
module orcus_operand_scratchpads #(
    parameter integer LANES = 32
) (
    input  wire clk,
    input  wire [$clog2(16384/(LANES*4))-1:0] wr_addr,
    input  wire a_we,
    input  wire b_we,
    input  wire [LANES*4-1:0] a_wdata,
    input  wire [LANES*4-1:0] b_wdata,
    input  wire rd_en,
    input  wire [$clog2(16384/(LANES*4))-1:0] rd_addr,
    output wire [LANES*4-1:0] a_rd_data,
    output wire [LANES*4-1:0] b_rd_data
);
    orcus_operand_scratchpad #(.LANES(LANES)) u_scratchpad_a (
        .clk(clk), .wr_en(a_we), .wr_addr(wr_addr), .wr_data(a_wdata),
        .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(a_rd_data)
    );
    orcus_operand_scratchpad #(.LANES(LANES)) u_scratchpad_b (
        .clk(clk), .wr_en(b_we), .wr_addr(wr_addr), .wr_data(b_wdata),
        .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(b_rd_data)
    );
endmodule


// Low-I/O harness for ECP5 placement of both 32-lane memories. The full
// scratchpad interface exceeds the selected package's user-I/O count.
module orcus_scratchpad_ecp5_top #(
    parameter integer LANES = 32
) (
    input wire clk,
    input wire enable,
    output wire status
);
    localparam integer WORD_WIDTH = LANES * 4;
    localparam integer DEPTH = 16384 / WORD_WIDTH;
    localparam integer ADDR_WIDTH = $clog2(DEPTH);

    reg [ADDR_WIDTH-1:0] wr_addr = 0;
    reg [ADDR_WIDTH-1:0] rd_addr = 1;
    reg [WORD_WIDTH-1:0] a_pattern = {WORD_WIDTH{1'b1}};
    reg [WORD_WIDTH-1:0] b_pattern = {{(WORD_WIDTH-1){1'b0}}, 1'b1};
    wire [WORD_WIDTH-1:0] a_rd_data;
    wire [WORD_WIDTH-1:0] b_rd_data;

    always @(posedge clk) begin
        if (enable) begin
            wr_addr <= wr_addr + 1'b1;
            // Read remains one address ahead of write, avoiding prohibited
            // same-address access for all supported power-of-two depths.
            rd_addr <= wr_addr + 2'd2;
            a_pattern <= {a_pattern[WORD_WIDTH-2:0],
                          a_pattern[WORD_WIDTH-1] ^ a_pattern[WORD_WIDTH-2] ^
                          a_pattern[WORD_WIDTH-5] ^ a_pattern[WORD_WIDTH-7]};
            b_pattern <= {b_pattern[WORD_WIDTH-2:0],
                          b_pattern[WORD_WIDTH-1] ^ b_pattern[WORD_WIDTH-3] ^
                          b_pattern[WORD_WIDTH-4] ^ b_pattern[WORD_WIDTH-6]};
        end
    end

    orcus_operand_scratchpads #(.LANES(LANES)) u_memories (
        .clk(clk), .wr_addr(wr_addr), .a_we(enable), .b_we(enable),
        .a_wdata(a_pattern), .b_wdata(b_pattern),
        .rd_en(enable), .rd_addr(rd_addr),
        .a_rd_data(a_rd_data), .b_rd_data(b_rd_data)
    );
    assign status = ^(a_rd_data ^ b_rd_data);
endmodule

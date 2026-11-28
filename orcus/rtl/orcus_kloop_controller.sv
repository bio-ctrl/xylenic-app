// ORCUS M4.3 — K-loop read issue and synchronous-read tag alignment.
// Tags are registered on the same edge as the M4.2 scratchpad read-data
// registers, so valid/first/last describe the rdata_* values after that edge.
module orcus_kloop_controller #(
    parameter integer LANES = 32
) (
    input  wire clk,
    input  wire reset,             // synchronous, active high
    input  wire start,
    input  wire [$clog2(16384/(LANES*4))-1:0] base_addr,
    input  wire [$clog2(16384/(LANES*4)+1)-1:0] k_length,

    // Host write activity is observed to enforce the v1 idle-only rule.
    input  wire [$clog2(16384/(LANES*4))-1:0] wr_addr,
    input  wire a_we,
    input  wire b_we,

    // Assert for one cycle when the downstream Dadda+accumulator pipeline
    // has committed the result for the active K-loop transaction.
    input  wire pipeline_done,

    output reg  busy,
    output wire rd_en,
    output wire [$clog2(16384/(LANES*4))-1:0] rd_addr,
    output reg  first,
    output reg  last,
    output reg  valid
);
    localparam integer MEMORY_BITS = 16384;
    localparam integer WORD_WIDTH = LANES * 4;
    localparam integer DEPTH = MEMORY_BITS / WORD_WIDTH;
    localparam integer ADDR_WIDTH = $clog2(DEPTH);
    localparam integer K_WIDTH = $clog2(DEPTH + 1);

    reg [ADDR_WIDTH-1:0] request_base;
    reg [K_WIDTH-1:0] request_length;
    reg [K_WIDTH-1:0] issued_count;
    reg all_reads_issued;

    wire [ADDR_WIDTH:0] base_ext = {1'b0, base_addr};
    wire [ADDR_WIDTH:0] k_ext =
        {{(ADDR_WIDTH+1-K_WIDTH){1'b0}}, k_length};
    wire [ADDR_WIDTH:0] request_end = base_ext + k_ext;
    wire request_in_bounds = (k_length != 0) &&
                             (base_addr < DEPTH) &&
                             (request_end <= DEPTH);

    // A new request may be accepted while idle, or on the exact completion
    // edge of the previous transaction. This permits zero-extra-bubble
    // transaction chaining without interleaving accumulation contexts.
    wire accept_window = !busy || (pipeline_done && all_reads_issued);
    wire host_start_collision = (a_we || b_we) && (wr_addr == base_addr);
    wire request_accept = !reset && start && accept_window &&
                          request_in_bounds && !host_start_collision;

    assign rd_en = !reset &&
                   (request_accept || (busy && !all_reads_issued));
    assign rd_addr = request_accept
                   ? base_addr
                   : request_base + issued_count[ADDR_WIDTH-1:0];

    wire [K_WIDTH-1:0] issue_index = request_accept
                                    ? {K_WIDTH{1'b0}}
                                    : issued_count;
    wire [K_WIDTH-1:0] active_length = request_accept
                                      ? k_length
                                      : request_length;

    always @(posedge clk) begin
        if (reset) begin
            busy             <= 1'b0;
            request_base     <= {ADDR_WIDTH{1'b0}};
            request_length   <= {K_WIDTH{1'b0}};
            issued_count     <= {K_WIDTH{1'b0}};
            all_reads_issued <= 1'b0;
            first            <= 1'b0;
            last             <= 1'b0;
            valid            <= 1'b0;
        end else begin
            // These three flops advance with the scratchpad's registered
            // read data. They are zero on every cycle without a read issue.
            valid <= rd_en;
            first <= rd_en && (issue_index == 0);
            last  <= rd_en && (issue_index == active_length - 1'b1);

            if (busy && (a_we || b_we))
                $fatal(1, "ORCUS host writes are prohibited while K-loop busy");
            if (pipeline_done && busy && !all_reads_issued)
                $fatal(1, "ORCUS pipeline_done arrived before final K read");
            if (rd_en && (a_we || b_we) && (wr_addr == rd_addr))
                $fatal(1, "ORCUS K-loop read/write address collision");

            if (request_accept) begin
                busy             <= 1'b1;
                request_base     <= base_addr;
                request_length   <= k_length;
                issued_count     <= {{(K_WIDTH-1){1'b0}}, 1'b1};
                all_reads_issued <= (k_length == 1);
            end else if (busy && !all_reads_issued) begin
                issued_count <= issued_count + 1'b1;
                if (issued_count == request_length - 1'b1)
                    all_reads_issued <= 1'b1;
            end else if (busy && pipeline_done && all_reads_issued) begin
                busy             <= 1'b0;
                issued_count     <= {K_WIDTH{1'b0}};
                all_reads_issued <= 1'b0;
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (!(LANES == 4 || LANES == 8 || LANES == 16 || LANES == 32))
            $fatal(1, "ORCUS K-loop LANES must be 4, 8, 16, or 32");
        if (DEPTH * WORD_WIDTH != MEMORY_BITS)
            $fatal(1, "ORCUS K-loop geometry must contain 16384 bits");
        if (K_WIDTH != $clog2(DEPTH + 1))
            $fatal(1, "ORCUS K-loop length width cannot represent full depth");
    end
`endif
endmodule

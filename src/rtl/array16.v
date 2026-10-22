`timescale 1ns/1ps
// Sixteen independent signed Q8.8 lanes; lane 0 occupies the lowest bits.
// Wiring-only wrapper: accepted edge k -> output edge k+9, II=1 per lane.
module array16 (
    input  wire clk,
    input  wire reset,
    input  wire [15:0] valid_in,
    input  wire [31:0] mode_in,
    input  wire [255:0] x_in,
    output wire [15:0] valid_out,
    output wire [31:0] mode_out,
    output wire [255:0] y_out
);
    genvar lane;
    generate
        for (lane = 0; lane < 16; lane = lane + 1) begin : lanes
            wrapper1 wrapper1 (
                .clk(clk),
                .reset(reset),
                .valid_in(valid_in[lane]),
                .mode_in(mode_in[2*lane +: 2]),
                .x_in(x_in[16*lane +: 16]),
                .valid_out(valid_out[lane]),
                .mode_out(mode_out[2*lane +: 2]),
                .y_out(y_out[16*lane +: 16])
            );
        end
    endgenerate
endmodule

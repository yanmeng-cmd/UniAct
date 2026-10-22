`timescale 1ns/1ps
// C2 dual-use: coefficient[2] selects the fitted correction branch directly.
// No run-time direction comparator/table is added; Z<8 retains its bypass.
module reciprocal (
    input wire [7:0] exponent_code,
    input wire [3:0] coefficient,
    output wire [7:0] correction
);
    assign correction = (exponent_code < 8'd8) ? exponent_code :
                        {2'd0, coefficient, coefficient[2], 1'b0};
endmodule

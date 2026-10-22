`timescale 1ns/1ps
// E13 / 2^(n+5), positive nearest rounding with ties away, then saturate.
module exp_round (
    input  wire [12:0] exponent_fraction,
    input  wire [4:0]  exponent_integer,
    output reg  [7:0]  exponent_code
);
    reg [13:0] rounded;
    always @* begin
        rounded = 14'd0;
        exponent_code = 8'd0;
        // n >= 9 is an explicit zero tail, not a large dynamic shift.
        if (exponent_integer < 5'd9) begin
            rounded = ({1'b0, exponent_fraction} +
                       (14'd16 << exponent_integer)) >>
                       (exponent_integer + 5'd5);
            exponent_code = (rounded > 14'd255) ? 8'd255 : rounded[7:0];
        end
    end
endmodule

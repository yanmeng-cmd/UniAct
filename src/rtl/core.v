`timescale 1ns/1ps
// Q4.4 core. Mode 00=sigmoid, 01=SiLU, 10=logistic GELU, 11=invalid.
// Eight register stages; accepted edge k -> valid output edge k+7.
// Output is signed P_DSP (scale 8192) with Q8.8 round-away bias already added.
module core (
    input  wire clk,
    input  wire reset,
    input  wire valid_in,
    input  wire [1:0] mode_in,
    input  wire signed [7:0] x_in,
    output wire valid_out,
    output wire [1:0] mode_out,
    output wire signed [16:0] result_out
);
    reg [7:0] valid_pipe;
    reg [1:0] mode_pipe [0:7];
    reg [7:0] u0, u1, u2, u3;
    reg sign0, sign1, sign2, sign3;
    reg [16:0] product1;
    reg [12:0] fraction2;
    reg [4:0] integer2;
    reg [7:0] z3;
    wire [16:0] u_extended = {9'd0, u0};
    wire [16:0] product_s = (u_extended << 8) + (u_extended << 7) -
                            (u_extended << 4) + u_extended;
    wire [16:0] product_g = (u_extended << 9) + (u_extended << 7) -
                            (u_extended << 4) + (u_extended << 2);
    wire [3:0] ce, cr;
    wire [13:0] fraction_next = 14'd8192 - {2'd0, product1[11:0]} -
                               {5'd0, ce, product1[6], 4'd0};
    wire [7:0] z_next;
    // D={Cr,Cr[2],0} on the normal path. Cr[2] is jointly fitted as a
    // coefficient bit and soft direction selector; no direction comparator.
    // Z<8 keeps the original D=Z bypass.
    wire [7:0] correction;
    wire [7:0] multiplier = (mode_pipe[3] == 2'd0) ? 8'd16 : u3;
    integer i;

    lut tables (
        .exponent_address(product1[11:6]), .reciprocal_address(z3[7:2]),
        .exponent_coefficient(ce), .reciprocal_coefficient(cr));
    exp_round exponent_round (
        .exponent_fraction(fraction2), .exponent_integer(integer2),
        .exponent_code(z_next));
    reciprocal reciprocal (
        .exponent_code(z3), .coefficient(cr), .correction(correction));
    dsp backend (
        .clk(clk), .reset(reset), .exponent_code(z3),
        .reciprocal_correction(correction), .magnitude(multiplier),
        .negative(sign3), .sigmoid(mode_pipe[3] == 2'd0), .result(result_out));

    always @(posedge clk) begin
        if (reset) begin
            valid_pipe <= 8'd0;
            for (i=0; i<8; i=i+1) mode_pipe[i] <= 2'd0;
            u0 <= 0; u1 <= 0; u2 <= 0; u3 <= 0;
            sign0 <= 0; sign1 <= 0; sign2 <= 0; sign3 <= 0;
            product1 <= 0; fraction2 <= 0; integer2 <= 0; z3 <= 0;
        end else begin
            valid_pipe <= {valid_pipe[6:0], valid_in && mode_in != 2'b11};
            mode_pipe[0] <= mode_in;
            for (i=1; i<8; i=i+1) mode_pipe[i] <= mode_pipe[i-1];
            // Widen before negation: the magnitude of -128 is unsigned 128.
            u0 <= x_in[7] ? -{x_in[7], x_in} : {1'b0, x_in};
            sign0 <= x_in[7];
            product1 <= (mode_pipe[0] == 2'd2) ? product_g : product_s;
            u1 <= u0; sign1 <= sign0;
            fraction2 <= fraction_next[12:0]; integer2 <= product1[16:12];
            u2 <= u1; sign2 <= sign1;
            z3 <= z_next; u3 <= u2; sign3 <= sign2;
        end
    end
    assign valid_out = valid_pipe[7];
    assign mode_out = mode_pipe[7];
endmodule

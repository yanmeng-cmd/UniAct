`timescale 1ns/1ps
// C2 dual-use soft-direction candidate v3: split-address fabric unchanged.
module lut (
    input  wire [5:0] exponent_address,
    input  wire [5:0] reciprocal_address,
    output wire [3:0] exponent_coefficient,
    output wire [3:0] reciprocal_coefficient
);
    localparam [63:0] LOW_INIT  = 64'hA74DDE0F59CAF5CA;
    localparam [63:0] HIGH_INIT = 64'h0A5EFFC00DA5003D;

    // Relative placement binds only the four coefficient cells to one slice;
    // it does not select a board-specific absolute SLICE location.
    (* HU_SET="activation_tables", RLOC="X0Y0", BEL="A6LUT" *)
    LUT6_2 #(.INIT(LOW_INIT)) exponent_low (
        .O5(exponent_coefficient[0]), .O6(exponent_coefficient[1]),
        .I0(exponent_address[0]), .I1(exponent_address[1]),
        .I2(exponent_address[2]), .I3(exponent_address[4]),
        .I4(exponent_address[5]), .I5(1'b1));
    (* HU_SET="activation_tables", RLOC="X0Y0", BEL="B6LUT" *)
    LUT6_2 #(.INIT(HIGH_INIT)) exponent_high (
        .O5(exponent_coefficient[2]), .O6(exponent_coefficient[3]),
        .I0(exponent_address[1]), .I1(exponent_address[2]),
        .I2(exponent_address[3]), .I3(exponent_address[4]),
        .I4(exponent_address[5]), .I5(1'b1));
    (* HU_SET="activation_tables", RLOC="X0Y0", BEL="C6LUT" *)
    LUT6_2 #(.INIT(LOW_INIT)) reciprocal_low (
        .O5(reciprocal_coefficient[0]), .O6(reciprocal_coefficient[1]),
        .I0(reciprocal_address[0]), .I1(reciprocal_address[1]),
        .I2(reciprocal_address[2]), .I3(reciprocal_address[4]),
        .I4(reciprocal_address[5]), .I5(1'b1));
    (* HU_SET="activation_tables", RLOC="X0Y0", BEL="D6LUT" *)
    LUT6_2 #(.INIT(HIGH_INIT)) reciprocal_high (
        .O5(reciprocal_coefficient[2]), .O6(reciprocal_coefficient[3]),
        .I0(reciprocal_address[1]), .I1(reciprocal_address[2]),
        .I2(reciprocal_address[3]), .I3(reciprocal_address[4]),
        .I4(reciprocal_address[5]), .I5(1'b1));
endmodule

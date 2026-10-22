`timescale 1ns/1ps
// Four register stages: A/D/B1 -> AD/B2 -> M/C/ALUMODE -> P.
// Input captured on edge k produces P on edge k+3; II=1.
module dsp (
    input  wire clk,
    input  wire reset,
    input  wire [7:0] exponent_code,
    input  wire [7:0] reciprocal_correction,
    input  wire [7:0] magnitude,
    input  wire negative,
    input  wire sigmoid,
    output wire signed [16:0] result
);
    // Add the exact Q8.8 round-away bias inside the DSP post-adder.
    // Nonnegative output: +16; negative SiLU/GELU output: +15.
    wire [47:0] negative_constant = sigmoid ? 48'd16 : 48'd15;
    wire [47:0] constant_term = negative ? negative_constant :
                                  ({31'd0, magnitude, 9'd0} + 48'd16);
    wire [3:0] alu_mode = (negative && sigmoid) ? 4'b0000 : 4'b0011;
    reg [47:0] constant_d1, constant_d2;
    reg [3:0] alu_d1, alu_d2;
    wire [47:0] dsp_result;
    always @(posedge clk) begin
        if (reset) begin
            constant_d1 <= 48'd0; constant_d2 <= 48'd0;
            alu_d1 <= 4'd0; alu_d2 <= 4'd0;
        end else begin
            constant_d1 <= constant_term; constant_d2 <= constant_d1;
            alu_d1 <= alu_mode; alu_d2 <= alu_d1;
        end
    end

    // Preadder: E + D; multiplier: M*(E + D); ALU: C +/- product.
    DSP48E2 #(
        .AREG(1), .ACASCREG(1), .DREG(1), .ADREG(1),
        .BREG(2), .BCASCREG(2), .MREG(1), .PREG(1),
        .CREG(1), .ALUMODEREG(1), .INMODEREG(0), .OPMODEREG(0),
        .CARRYINREG(0), .CARRYINSELREG(0),
        .AMULTSEL("AD"), .PREADDINSEL("A"),
        .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48")
    ) arithmetic (
        .CLK(clk), .A({22'd0, exponent_code}),
        .D({19'd0, reciprocal_correction}), .B({10'd0, magnitude}),
        .C(constant_d2), .P(dsp_result),
        // X=U, Y=V, Z=C selects C +/- the multiplier result.
        .INMODE(5'b00100), .OPMODE(9'b000110101), .ALUMODE(alu_d2),
        .ACIN(30'd0), .BCIN(18'd0), .PCIN(48'd0),
        .CARRYIN(1'b0), .CARRYCASCIN(1'b0), .CARRYINSEL(3'd0),
        .MULTSIGNIN(1'b0),
        .CEA1(1'b1), .CEA2(1'b1), .CED(1'b1), .CEAD(1'b1),
        .CEB1(1'b1), .CEB2(1'b1), .CEM(1'b1), .CEP(1'b1),
        .CEC(1'b1), .CEALUMODE(1'b1), .CECTRL(1'b1),
        .CEINMODE(1'b1), .CECARRYIN(1'b1),
        .RSTA(reset), .RSTD(reset), .RSTB(reset), .RSTM(reset), .RSTP(reset),
        .RSTC(reset), .RSTALUMODE(reset), .RSTCTRL(reset),
        .RSTINMODE(reset), .RSTALLCARRYIN(reset),
        .ACOUT(), .BCOUT(), .PCOUT(), .CARRYOUT(), .CARRYCASCOUT(),
        .MULTSIGNOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(),
        .PATTERNBDETECT(), .XOROUT());
    assign result = dsp_result[16:0];
endmodule

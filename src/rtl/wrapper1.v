`timescale 1ns/1ps
// External signed Q8.8 -> signed Q8.8. Valid-only streaming, no backpressure.
// Ten register stages; accepted edge k -> valid output edge k+9. II=1.
module wrapper1 (
    input  wire clk,
    input  wire reset,
    input  wire valid_in,
    input  wire [1:0] mode_in,
    input  wire signed [15:0] x_in,
    output reg valid_out,
    output reg [1:0] mode_out,
    output reg signed [15:0] y_out
);
    wire outside_signed12 = x_in[15:12] != {4{x_in[11]}};
    wire positive_tail = !x_in[15] &&
                         (outside_signed12 || x_in[11:4] == 8'h7f);
    wire negative_tail = x_in[15] &&
                         (outside_signed12 || (x_in[11:4] == 8'h80 && x_in[3:0] == 0));
    wire [16:0] input_magnitude = x_in[15] ? -{x_in[15], x_in} : {1'b0,x_in};
    wire [16:0] rounded_input_magnitude = (input_magnitude + 17'd8) >> 4;
    reg signed [7:0] core_x;
    reg core_valid;
    reg [1:0] core_mode;
    reg [1:0] bypass_pipe [0:8]; // 00=core, 01=negative tail, 10=positive tail
    reg signed [15:0] raw_pipe [0:8];
    wire core_valid_out;
    wire [1:0] core_mode_out;
    wire signed [16:0] core_result;
    // DSP C already contains the sign-specific round-away bias.
    wire signed [15:0] rounded_result = core_result >>> 5;
    integer i;
    core core (
        .clk(clk), .reset(reset), .valid_in(core_valid), .mode_in(core_mode),
        .x_in(core_x), .valid_out(core_valid_out), .mode_out(core_mode_out),
        .result_out(core_result));
    always @(posedge clk) begin
        if (reset) begin
            core_x <= 0; core_valid <= 0; core_mode <= 0;
            valid_out <= 0; mode_out <= 0; y_out <= 0;
            for (i=0; i<9; i=i+1) begin bypass_pipe[i] <= 0; raw_pipe[i] <= 0; end
        end else begin
            core_valid <= valid_in && mode_in != 2'b11;
            core_mode <= mode_in;
            core_x <= (positive_tail || negative_tail) ? 8'sd0 :
                      (x_in[15] ? -rounded_input_magnitude[7:0] : rounded_input_magnitude[7:0]);
            bypass_pipe[0] <= negative_tail ? 2'd1 : (positive_tail ? 2'd2 : 2'd0);
            raw_pipe[0] <= x_in;
            for (i=1; i<9; i=i+1) begin
                bypass_pipe[i] <= bypass_pipe[i-1]; raw_pipe[i] <= raw_pipe[i-1];
            end
            valid_out <= core_valid_out;
            mode_out <= core_mode_out;
            case (bypass_pipe[8])
                2'd1: y_out <= 16'sd0;
                2'd2: y_out <= (core_mode_out == 2'd0) ? 16'sd256 : raw_pipe[8];
                default: y_out <= rounded_result;
            endcase
        end
    end
endmodule

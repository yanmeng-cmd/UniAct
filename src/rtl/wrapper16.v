`timescale 1ns/1ps
// Two accepted 128-bit beats form one 16-lane Q8.8 batch.
// ready_in is an output to the producer; ready_out is an input from the consumer.
// First input beat: lanes 0..7; second: lanes 8..15. Output uses the same order.
module wrapper16 (
    input  wire clk,
    input  wire reset,
    input  wire valid_in,
    output wire ready_in,
    input  wire [7:0] lane_valid_in,
    input  wire [15:0] mode_in,
    input  wire [127:0] x_in,
    output wire valid_out,
    input  wire ready_out,
    output wire half_out,
    output wire [7:0] lane_valid_out,
    output wire [15:0] mode_out,
    output wire [127:0] y_out
);
    // Reserve a slot at batch launch, including its time in the compute pipeline.
    localparam integer SLOTS = 8;
    reg input_half;
    reg [127:0] first_x;
    reg [15:0] first_mode;
    reg [7:0] first_valid;
    reg [3:0] reserved;
    reg [2:0] write_pointer, read_pointer;
    reg output_half;
    reg [SLOTS-1:0] completed;
    reg [255:0] results [0:SLOTS-1];
    reg [31:0] result_modes [0:SLOTS-1];
    reg [15:0] result_valids [0:SLOTS-1];
    // Array output updates at launch+9; capture it at launch+10.
    reg [9:0] token_valid;
    reg [2:0] token_slot [0:9];
    wire [15:0] array_valid;
    wire [31:0] array_mode;
    wire [255:0] array_y;
    wire take_output = valid_out && ready_out;
    wire retire = take_output && output_half;
    wire take_input = valid_in && ready_in;
    wire launch = take_input && input_half;
    integer stage;

    assign valid_out = !reset && completed[read_pointer];
    // A freed credit becomes visible next cycle; no output-to-input handshake feedback.
    assign ready_in = !reset && (reserved < SLOTS);
    assign half_out = output_half;
    assign y_out = output_half ? results[read_pointer][255:128] :
                                results[read_pointer][127:0];
    assign mode_out = output_half ? result_modes[read_pointer][31:16] :
                                   result_modes[read_pointer][15:0];
    assign lane_valid_out = output_half ? result_valids[read_pointer][15:8] :
                                         result_valids[read_pointer][7:0];

    array16 array (
        .clk(clk), .reset(reset),
        .valid_in({16{launch}} & {lane_valid_in, first_valid}),
        .mode_in({mode_in, first_mode}), .x_in({x_in, first_x}),
        .valid_out(array_valid), .mode_out(array_mode), .y_out(array_y)
    );

    always @(posedge clk) begin
        if (reset) begin
            input_half <= 1'b0;
            first_x <= 0; first_mode <= 0; first_valid <= 0;
            reserved <= 0;
            write_pointer <= 0; read_pointer <= 0;
            output_half <= 1'b0;
            completed <= 0;
            token_valid <= 0;
            for (stage = 0; stage < 10; stage = stage + 1)
                token_slot[stage] <= 0;
        end else begin
            if (take_input) begin
                input_half <= !input_half;
                if (!input_half) begin
                    first_x <= x_in;
                    first_mode <= mode_in;
                    first_valid <= lane_valid_in;
                end
            end
            if (launch) write_pointer <= write_pointer + 3'd1;
            case ({launch, retire})
                2'b10: reserved <= reserved + 4'd1;
                2'b01: reserved <= reserved - 4'd1;
                default: reserved <= reserved;
            endcase
            token_valid <= {token_valid[8:0], launch};
            token_slot[0] <= write_pointer;
            for (stage = 1; stage < 10; stage = stage + 1)
                token_slot[stage] <= token_slot[stage-1];
            if (take_output) begin
                output_half <= !output_half;
                if (output_half) begin
                    completed[read_pointer] <= 1'b0;
                    read_pointer <= read_pointer + 3'd1;
                end
            end
            if (token_valid[9]) begin
                results[token_slot[9]] <= array_y;
                result_modes[token_slot[9]] <= array_mode;
                result_valids[token_slot[9]] <= array_valid;
                completed[token_slot[9]] <= 1'b1;
            end
        end
    end
endmodule

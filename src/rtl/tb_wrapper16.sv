`timescale 1ns/1ps
module tb_wrapper16;
    localparam integer CASES = 196608;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset = 1, valid_in = 0, ready_out = 1;
    wire ready_in, valid_out, half_out;
    reg [7:0] lane_valid_in = 0;
    reg [15:0] mode_in = 0;
    reg [127:0] x_in = 0;
    wire [7:0] lane_valid_out;
    wire [15:0] mode_out;
    wire [127:0] y_out;
    reg [33:0] vectors [0:CASES-1];
    reg [15:0] golden [0:CASES-1];
    reg [152:0] expected [0:65535];
    integer launch_cycle [0:32767];
    integer write_count = 0, read_count = 0, cycle = 0;
    integer checked = 0, beats_checked = 0, input_stalls = 0, output_stalls = 0;
    integer consecutive = 0, max_consecutive = 0;
    integer index, mode, lane, half, batch, address, monitor_lane;
    integer base_checked, base_beats;
    integer boundaries [0:15];
    reg input_half = 0, held = 0, check_latency = 0;
    reg [152:0] held_payload;
    reg [152:0] reference_beat;
    reg [127:0] drive_x;
    reg [15:0] drive_mode;
    reg [7:0] drive_mask;
    reg [31:0] data_random = 32'h16c22026;
    reg [31:0] ready_random = 32'h6420bad1;
    integer test_x, test_mode, test_y;

    // Independent integer reference; no Python or external vector files.
    // Packed 2-bit table entries: address 0 occupies bits [1:0].
    localparam [63:0] TABLE_L = 64'h996B70E6F7B950EE;
    localparam [63:0] TABLE_H = 64'h00D966B9AAAAA551;

    function automatic integer ref_coefficient(input integer j);
        integer al, ah;
        begin
            al = (j & 7) | ((j >> 4) << 3);
            ah = j >> 1;
            ref_coefficient = ((TABLE_L >> (2*al)) & 3) +
                              4*((TABLE_H >> (2*ah)) & 3);
        end
    endfunction

    // Unbiased core output, with real-value scale 8192.
    function automatic integer ref_core(input integer xq, input integer mode);
        integer u, p, n, t, ef, z, cr, d, m;
        begin
            u = (xq < 0) ? -xq : xq;
            p = u * ((mode == 2) ? 628 : 369);
            n = p >> 12;
            t = p & 4095;
            ef = 8192-t-32*ref_coefficient(t >> 6)-16*((t >> 6) & 1);
            z = (n >= 9) ? 0 : ((ef+(1 << (n+4))) >> (n+5));
            if (z > 255) z = 255;
            cr = ref_coefficient(z >> 2);
            d = (z < 8) ? z : 4*cr+2*((cr >> 2) & 1);
            m = (mode == 0) ? 16 : u;
            if (xq >= 0) ref_core = m*(512-z-d);
            else if (mode == 0) ref_core = m*(z+d);
            else ref_core = -m*(z+d);
        end
    endfunction

    // Signed Q8.8 input/output with range bypass and round-away.
    function automatic integer ref_external(input integer x, input integer mode);
        integer xq, pout;
        begin
            if (x <= -2048) ref_external = 0;
            else if (x >= 2032) ref_external = (mode == 0) ? 256 : x;
            else begin
                xq = (x < 0) ? -((-x+8) >> 4) : ((x+8) >> 4);
                pout = ref_core(xq, mode);
                ref_external = (pout < 0) ? -((-pout+16) >> 5) : ((pout+16) >> 5);
            end
        end
    endfunction

    wrapper16 dut (
        .clk(clk), .reset(reset), .valid_in(valid_in), .ready_in(ready_in),
        .lane_valid_in(lane_valid_in), .mode_in(mode_in), .x_in(x_in),
        .valid_out(valid_out), .ready_out(ready_out), .half_out(half_out),
        .lane_valid_out(lane_valid_out), .mode_out(mode_out), .y_out(y_out)
    );

    // Queue transactions only on a handshake. Check ordering, masks, modes,
    // numeric results, stalls and externally observable fixed no-stall timing.
    always @(posedge clk) begin
        cycle = cycle + 1;
        if (reset) begin
            write_count = 0; read_count = 0; input_half = 0;
            held = 0; consecutive = 0;
        end else begin
            if (valid_in && ready_in) begin
                reference_beat = 0;
                reference_beat[152] = input_half;
                reference_beat[143:128] = mode_in;
                for (monitor_lane = 0; monitor_lane < 8; monitor_lane = monitor_lane + 1) begin
                    reference_beat[144+monitor_lane] = lane_valid_in[monitor_lane] &&
                        mode_in[2*monitor_lane +: 2] != 3;
                    if (reference_beat[144+monitor_lane])
                        reference_beat[16*monitor_lane +: 16] = golden[
                            mode_in[2*monitor_lane +: 2]*65536 + x_in[16*monitor_lane +: 16]];
                end
                expected[write_count] = reference_beat;
                if (input_half) launch_cycle[write_count/2] = cycle;
                input_half = !input_half;
                write_count = write_count + 1;
            end
            if (valid_in && !ready_in) input_stalls = input_stalls + 1;
            if (held && (!valid_out ||
                {half_out,lane_valid_out,mode_out,y_out} !== held_payload))
                $fatal(1, "output changed before stalled beat was accepted");
            held = valid_out && !ready_out;
            held_payload = {half_out,lane_valid_out,mode_out,y_out};
            if (held) output_stalls = output_stalls + 1;
            if (valid_out && ready_out) begin
                if (read_count >= write_count) $fatal(1, "unexpected or post-reset output");
                reference_beat = expected[read_count];
                if ({half_out,lane_valid_out,mode_out} !== reference_beat[152:128])
                    $fatal(1, "half/mask/mode mismatch beat=%0d", read_count);
                for (monitor_lane = 0; monitor_lane < 8; monitor_lane = monitor_lane + 1) begin
                    if (reference_beat[144+monitor_lane]) begin
                        if (y_out[16*monitor_lane +: 16] !== reference_beat[16*monitor_lane +: 16])
                            $fatal(1, "numeric mismatch beat=%0d lane=%0d expected=%h actual=%h",
                                read_count, monitor_lane, reference_beat[16*monitor_lane +: 16],
                                y_out[16*monitor_lane +: 16]);
                        checked = checked + 1;
                    end
                end
                if (check_latency && cycle-launch_cycle[read_count/2] != 11+half_out)
                    $fatal(1, "no-stall latency mismatch: %0d", cycle-launch_cycle[read_count/2]);
                read_count = read_count + 1;
                beats_checked = beats_checked + 1;
                consecutive = consecutive + 1;
                if (consecutive > max_consecutive) max_consecutive = consecutive;
            end else consecutive = 0;
        end
        #1;
        if (reset && (valid_out !== 0 || ready_in !== 0))
            $fatal(1, "handshake active during reset");
    end

    task send_beat(input [127:0] data, input [15:0] modes, input [7:0] mask);
        begin
            @(negedge clk);
            valid_in = 1; x_in = data; mode_in = modes; lane_valid_in = mask;
            do @(posedge clk); while (!ready_in);
        end
    endtask

    task drain;
        begin
            @(negedge clk); valid_in = 0; ready_out = 1;
            while (read_count != write_count) @(negedge clk);
            repeat (14) @(negedge clk);
            if (valid_out !== 0) $fatal(1, "extra output after drain");
        end
    endtask

    task reset_pipeline;
        begin
            @(negedge clk); reset = 1; valid_in = 0;
            repeat (3) @(negedge clk);
            reset = 0; ready_out = 1;
        end
    endtask

    initial begin
        for (index = 0; index < CASES; index = index + 1) begin
            test_x=(((index/3)*40503) & 65535)-32768; test_mode=index%3;
            test_y=ref_external(test_x,test_mode);
            vectors[index]={test_x[15:0],test_mode[1:0],test_y[15:0]};
            address=test_mode*65536+(test_x & 65535);
            golden[address]=test_y;
        end
        boundaries[0] = -32768; boundaries[1] = -2049;
        boundaries[2] = -2048; boundaries[3] = -2047;
        boundaries[4] = -129; boundaries[5] = -8;
        boundaries[6] = -1; boundaries[7] = 0;
        boundaries[8] = 1; boundaries[9] = 7;
        boundaries[10] = 8; boundaries[11] = 129;
        boundaries[12] = 2031; boundaries[13] = 2032;
        boundaries[14] = 2033; boundaries[15] = 32767;
        #110;
        reset_pipeline();
        // Reset discards an unmatched first input beat.
        send_beat(0, 0, 8'hff);
        reset_pipeline();
        drain();
        // Reset discards batches still inside the computation pipeline.
        repeat (4) send_beat(0, 0, 8'hff);
        reset_pipeline();
        drain();

        for (batch = 0; batch < 16; batch = batch + 1) begin
            for (mode = 0; mode < 3; mode = mode + 1) begin
                for (half = 0; half < 2; half = half + 1) begin
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        drive_x[16*lane +: 16] = boundaries[(batch+8*half+lane)%16];
                        drive_mode[2*lane +: 2] = (mode+8*half+lane)%3;
                    end
                    send_beat(drive_x, drive_mode, 8'hff);
                end
            end
        end
        drain();
        base_checked = checked; base_beats = beats_checked;
        check_latency = 1;
        // Complete golden grid: 8 elements per beat, 16 per batch.
        for (batch = 0; batch < CASES/8; batch = batch + 1) begin
            for (lane = 0; lane < 8; lane = lane + 1) begin
                index = 8*batch + lane;
                drive_x[16*lane +: 16] = vectors[index][33:18];
                drive_mode[2*lane +: 2] = vectors[index][17:16];
            end
            send_beat(drive_x, drive_mode, 8'hff);
        end
        drain();
        check_latency = 0;
        if (checked-base_checked != CASES || beats_checked-base_beats != CASES/8 ||
            max_consecutive < CASES/8)
            $fatal(1, "incomplete grid or failure to sustain 8 elements/clock");

        // Stop the consumer longer than the core pipeline: fill all credits,
        // stall the producer, then release it and wrap the slot pointers.
        fork
            begin
                for (batch = 0; batch < 96; batch = batch + 1) begin
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        drive_x[16*lane +: 16] = (batch*125+lane*17)%8192-4096;
                        drive_mode[2*lane +: 2] = (batch+lane)%3;
                    end
                    send_beat(drive_x, drive_mode, 8'hff);
                end
                @(negedge clk); valid_in = 0;
            end
            begin
                @(negedge clk); ready_out = 0;
                repeat (60) @(negedge clk);
                ready_out = 1;
            end
        join
        drain();
        if (input_stalls == 0 || output_stalls == 0) $fatal(1, "backpressure not exercised");
        // All-invalid batches still produce two transport beats.
        send_beat(0, 16'hffff, 8'hff);
        send_beat(0, 0, 0);
        drain();
        fork
            begin
                for (batch = 0; batch < 2048; batch = batch + 1) begin
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        data_random = {data_random[30:0], data_random[31] ^
                            data_random[21] ^ data_random[1] ^ data_random[0]};
                        drive_x[16*lane +: 16] = data_random[15:0];
                        drive_mode[2*lane +: 2] = data_random[17:16];
                        drive_mask[lane] = data_random[18];
                    end
                    if (batch%7 == 0) begin @(negedge clk); valid_in = 0; end
                    send_beat(drive_x, drive_mode, drive_mask);
                end
                @(negedge clk); valid_in = 0;
            end
            begin
                repeat (4096) begin
                    @(negedge clk);
                    ready_random = {ready_random[30:0], ready_random[31] ^ ready_random[21] ^
                        ready_random[1] ^ ready_random[0]};
                    ready_out = ready_random[2:0] != 0;
                end
                ready_out = 1;
            end
        join
        drain();
        // Reset with a completed result held by a blocked consumer.
        @(negedge clk); ready_out = 0;
        repeat (4) send_beat(0, 0, 8'hff);
        repeat (16) @(negedge clk);
        if (!valid_out) $fatal(1, "expected a pending output before reset");
        reset_pipeline();
        drain();
        // Reset also discards a batch after its first output half was consumed.
        send_beat(0, 0, 8'hff); send_beat(0, 0, 8'hff);
        @(negedge clk); valid_in = 0;
        do @(posedge clk); while (!(valid_out && ready_out && !half_out));
        reset_pipeline();
        drain();
        $display("PASS: wrapper16_A lanes=16 golden_grid=%0d checked=%0d beats=%0d input_stalls=%0d output_stalls=%0d sustained=8_elements_per_cycle core_latency=9 input_boundary=0 first_beat_handshake=11 second_beat_handshake=12 masks=PASS invalid_modes=PASS reset_flush=PASS",
            CASES, checked, beats_checked, input_stalls, output_stalls);
        $finish;
    end
    initial begin #3000000; $fatal(1, "wrapper16 simulation watchdog timeout"); end
endmodule

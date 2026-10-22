`timescale 1ns/1ps
module tb_wrapper1;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset = 1;
    reg ext_valid = 0, core_valid = 0;
    reg [1:0] ext_mode = 0, core_mode = 0;
    reg signed [15:0] ext_x = 0;
    reg signed [7:0] core_x = 0;
    wire ext_vout, core_vout;
    wire [1:0] ext_mout, core_mout;
    wire signed [15:0] ext_y;
    wire signed [16:0] core_y;
    reg signed [15:0] ext_expected = 0;
    reg signed [16:0] core_expected = 0;
    reg ev [0:9], cv [0:7];
    reg [1:0] em [0:9], cm [0:7];
    reg signed [15:0] ey [0:9];
    reg signed [16:0] cy [0:7];
    integer ext_checked = 0, core_checked = 0, bubbles = 0, invalid_modes = 0;
    integer i, k, count, code, n, ref_z, lut_checked = 0, round_checked = 0;
    integer test_x, test_mode;
    reg [3:0] coefficients [0:63];
    reg [5:0] eaddr, raddr;
    wire [3:0] ce, cr;
    reg [12:0] fraction = 0;
    reg [4:0] exponent_integer = 0;
    wire [7:0] exponent_code;
    reg dsp_valid = 0, dsp_negative = 0, dsp_sigmoid = 0;
    reg [7:0] dsp_z = 0, dsp_magnitude = 0;
    wire [7:0] dsp_correction;
    wire signed [16:0] dsp_result;
    reg signed [16:0] dsp_expected = 0;
    reg dv [0:3];
    reg signed [16:0] dy [0:3];
    integer dsp_checked = 0, kind, ref_d, ref_s;

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

    wrapper1 dut (.clk(clk), .reset(reset), .valid_in(ext_valid),
        .mode_in(ext_mode), .x_in(ext_x), .valid_out(ext_vout), .mode_out(ext_mout), .y_out(ext_y));
    core core_dut (.clk(clk), .reset(reset), .valid_in(core_valid),
        .mode_in(core_mode), .x_in(core_x), .valid_out(core_vout), .mode_out(core_mout), .result_out(core_y));
    lut lut_dut (.exponent_address(eaddr), .reciprocal_address(raddr),
        .exponent_coefficient(ce), .reciprocal_coefficient(cr));
    exp_round round_dut (.exponent_fraction(fraction),
        .exponent_integer(exponent_integer), .exponent_code(exponent_code));
    reciprocal reciprocal_dut (.exponent_code(dsp_z),
        .coefficient(coefficients[dsp_z >> 2]), .correction(dsp_correction));
    dsp dsp_dut (.clk(clk), .reset(reset), .exponent_code(dsp_z),
        .reciprocal_correction(dsp_correction), .magnitude(dsp_magnitude),
        .negative(dsp_negative), .sigmoid(dsp_sigmoid), .result(dsp_result));

    // Independent transaction scoreboard: no output is trusted without checking
    // its expected input transaction, validity, mode and fixed edge latency.
    always @(posedge clk) begin
        if (reset) begin
            for (i=0; i<10; i=i+1) begin ev[i]=0; em[i]=0; ey[i]=0; end
            for (i=0; i<8; i=i+1) begin cv[i]=0; cm[i]=0; cy[i]=0; end
            for (i=0; i<4; i=i+1) begin dv[i]=0; dy[i]=0; end
        end else begin
            for (i=9; i>0; i=i-1) begin ev[i]=ev[i-1]; em[i]=em[i-1]; ey[i]=ey[i-1]; end
            for (i=7; i>0; i=i-1) begin cv[i]=cv[i-1]; cm[i]=cm[i-1]; cy[i]=cy[i-1]; end
            ev[0]=ext_valid && ext_mode != 3; em[0]=ext_mode; ey[0]=ext_expected;
            cv[0]=core_valid && core_mode != 3; cm[0]=core_mode; cy[0]=core_expected;
            for (i=3; i>0; i=i-1) begin dv[i]=dv[i-1]; dy[i]=dy[i-1]; end
            dv[0]=dsp_valid; dy[0]=dsp_expected;
        end
        #1;
        if (ext_vout !== ev[9]) $fatal(1,"external valid/reset/latency mismatch at %t",$time);
        if (core_vout !== cv[7]) $fatal(1,"core valid/reset/latency mismatch at %t",$time);
        if (ev[9]) begin
            if (ext_y !== ey[9] || ext_mout !== em[9])
                $fatal(1,"external mismatch: mode=%d expected=%d actual=%d at %t",em[9],ey[9],ext_y,$time);
            ext_checked=ext_checked+1;
        end
        if (cv[7]) begin
            if (core_y !== cy[7] || core_mout !== cm[7])
                $fatal(1,"core mismatch: mode=%d expected=%d actual=%d at %t",cm[7],cy[7],core_y,$time);
            core_checked=core_checked+1;
        end
        if (dv[3]) begin
            if (dsp_result !== dy[3])
                $fatal(1,"DSP mismatch expected=%d actual=%d",dy[3],dsp_result);
            dsp_checked=dsp_checked+1;
        end
    end

    task idle_cycle;
        begin @(negedge clk); ext_valid=0; core_valid=0; dsp_valid=0; bubbles=bubbles+1; end
    endtask
    task flush;
        begin repeat(12) idle_cycle(); end
    endtask
    task reset_pipeline;
        begin
            @(negedge clk); reset=1; ext_valid=0; core_valid=0; dsp_valid=0;
            repeat(3) @(negedge clk);
            reset=0;
        end
    endtask

    initial begin
        for (k=0; k<64; k=k+1) coefficients[k]=ref_coefficient(k);
        // The real Xilinx glbl model holds GSR for the first 100 ns.
        #110;
        // All 64 x 64 independent simultaneous coefficient addresses.
        for (k=0; k<4096; k=k+1) begin
            eaddr=k/64; raddr=k%64; #1;
            if (ce !== coefficients[eaddr] || cr !== coefficients[raddr])
                $fatal(1,"LUT mismatch e=%d r=%d ce=%h cr=%h expected=%h/%h",eaddr,raddr,ce,cr,coefficients[eaddr],coefficients[raddr]);
            lut_checked=lut_checked+1;
        end
        // All legal E13 values and every 5-bit n, including n=8/9,
        // saturation to 255 and zero-tail behavior.
        for (n=0; n<32; n=n+1) begin
            exponent_integer=n;
            for (code=1; code<8192; code=code+1) begin
                fraction=code;
                ref_z=(n>=9) ? 0 : ((code+(1<<(n+4)))>>(n+5));
                if (ref_z>255) ref_z=255;
                #1;
                if (exponent_code !== ref_z[7:0]) $fatal(1,"exponent round mismatch E=%d n=%d",code,n);
                round_checked=round_checked+1;
            end
        end
        reset_pipeline();
        // Reset while transactions occupy the pipeline: none may leak out.
        repeat(4) begin
            @(negedge clk); ext_valid=1; ext_mode=0; ext_x=0; ext_expected=129;
            core_valid=1; core_mode=0; core_x=0; core_expected=4144;
        end
        reset_pipeline();
        flush();
        // Full Z domain exercises Z=0/7/8/255 even when a value is not
        // reachable from the current Q4.4 input grid. Mix all ALU cases.
        for (code=0; code<256; code=code+1) begin
            for (kind=0; kind<4; kind=kind+1) begin
                @(negedge clk); dsp_valid=1; dsp_z=code;
                dsp_negative=kind[0]; dsp_sigmoid=kind[1];
                dsp_magnitude=dsp_sigmoid ? 16 : (dsp_negative ? 128 : 127);
                ref_d=(code<8) ? code : 4*coefficients[code>>2]+2*((coefficients[code>>2]>>2)&1);
                ref_s=code+ref_d;
                dsp_expected=dsp_negative ? (dsp_sigmoid ? dsp_magnitude*ref_s : -dsp_magnitude*ref_s) :
                                            dsp_magnitude*(512-ref_s);
                dsp_expected=dsp_expected +
                    ((dsp_negative && !dsp_sigmoid) ? 17'd15 : 17'd16);
                #1;
                if (dsp_correction !== ref_d[7:0]) $fatal(1,"reciprocal correction mismatch Z=%d",code);
            end
        end
        flush();
        if (dsp_checked!=1024) $fatal(1,"incomplete reciprocal/DSP cases %d",dsp_checked);
        count=0;
        for (k=0; k<768; k=k+1) begin
            @(negedge clk);
            test_x=-128+k/3; test_mode=k%3;
            core_x=test_x; core_mode=test_mode;
            core_expected=ref_core(test_x,test_mode)+((test_x<0 && test_mode!=0) ? 15 : 16);
            core_valid=1;
            count=count+1;
            if (count%19==0) idle_cycle();
            if (count%31==0) begin
                @(negedge clk); core_valid=1; core_mode=3; invalid_modes=invalid_modes+1;
            end
        end
        flush();
        if (core_checked!=768) $fatal(1,"incomplete core grid %d",core_checked);
        count=0;
        for (k=0; k<196608; k=k+1) begin
            @(negedge clk);
            test_x=(((k/3)*40503) & 65535)-32768; test_mode=k%3;
            ext_x=test_x; ext_mode=test_mode;
            ext_expected=ref_external(test_x,test_mode); ext_valid=1;
            count=count+1;
            if (count%997==0) idle_cycle();
            if (count%1013==0) begin
                @(negedge clk); ext_valid=1; ext_mode=3; invalid_modes=invalid_modes+1;
            end
        end
        flush();
        if (ext_checked!=196608) $fatal(1,"incomplete external grid %d",ext_checked);
        $display("PASS: LUT=%0d exponent_round=%0d reciprocal_DSP=%0d core=%0d external=%0d bubbles=%0d invalid_modes=%0d reset_flush=PASS II=1=PASS",
                 lut_checked,round_checked,dsp_checked,core_checked,ext_checked,bubbles,invalid_modes);
        $finish;
    end
    initial begin #10000000; $fatal(1,"simulation watchdog timeout"); end
endmodule

// https://www.chipverify.com/verilog/verilog-gate-level-modelin
// https://www.javatpoint.com/verilog-gate-level-modeling

module gates ( input a, b,
               output c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    
    // Multiple input  (out, in1, ..., inN);
    and can_name_if_wanted(c, a, b); 
    or(d, a, b);
    xor(e, a, b);
    nand(f, a, b);
    nor(g, a, b);
    xnor(h, a, a);

    // Multiple Output  (out1, out2, ..., out2, input); 
    buf(i, a);
    not(j, a);

    // Tristate (outputA, inputB, controlC); 
    bufif0(k, a, b);
    notif0(l, a, b);
    bufif1(m, a, b);
    notif1(n, a, b);
    
    // Pull gates
    pullup(o);
    pulldown(p);
endmodule

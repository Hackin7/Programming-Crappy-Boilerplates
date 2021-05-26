module m_not(output b, input a);
    nand(b,a,a);
endmodule

module m_and(output c, input a, input b);
    wire d;
    nand(d,a,b); not(c,d);
endmodule

module m_nor(output c, input a, input b);
    wire na, nb;
    // 1 0 0 0
    not(na,a); not(nb,b);
    and(c, na, nb);
endmodule

module m_or(output c, input a, input b);
    wire d;
    nor(d, a, b);
    not(c,d);
endmodule

module m_xor(output c, input a, input b);
    wire d, e;
    or(d, a,b); // 0 1 1 1
    nand(e, a, b); // 1 1 1 0
    and(c, d, e);
endmodule

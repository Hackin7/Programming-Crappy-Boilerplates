/*
Compilation Steps
  iverilog -o hello hello.v
  vvp hello
*/
module test;
  reg a, b;
  wire q,w,e,r,t,y;

  m_and z(q, a, b);
  m_or x(w, a, b);
  m_xor c(e, a, b);
  m_not v(r, a);
  nand(t, a, b);
  m_nor n(y, a, b);
  initial
    begin
      a=0;b=0;
      $monitor("%b %b %b %b %b %b",q,w,e,r,t,y);
      #10 a=1; b=0;
      #10 a=0; b=1;
      #10 a=1; b=1;
      $finish ;
    end
endmodule

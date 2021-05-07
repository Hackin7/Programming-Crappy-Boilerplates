/*
Compilation Steps
  iverilog -o hello hello.v
  vvp hello
*/
module test;
  reg S=0, R=0;
  wire Q, Qn;
  sr_flip_flop s(Q, Qn, S, R);

  initial begin
    $monitor("S: %b, R: %b, Q: %b, Qn: %b", S, R, Q, Qn);
    $display("Hello, World");
    #1 S=1;
    #1 S=0;
    #1 R=1;
    #1 R=0;
    $finish;
  end
endmodule

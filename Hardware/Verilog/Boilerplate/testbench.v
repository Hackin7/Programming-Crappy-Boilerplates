/*
Compilation Steps
  iverilog -o hello hello.v
  vvp hello
*/
module test;
  initial
    begin
      $display("Hello, World");
      $finish ;
    end
endmodule

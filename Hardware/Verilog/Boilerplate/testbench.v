/*
Compilation Steps
  iverilog -o hello hello.v
  vvp hello
*/

`include "main.v"
module test;
  initial
    begin
      $display("Hello, World");
      $finish ;
    end
endmodule

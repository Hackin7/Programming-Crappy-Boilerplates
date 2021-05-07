module test;
  reg debug = 1;
  reg reset = 0;
  reg increment = 0;
  reg enable_out = 0;
  wire [7:0] value;

  integer i = 0;

  program_counter pc (debug, reset, increment, enable_out, value);
  
  always #2 increment = !increment;
  always #512 reset = 1;
  initial begin
    $monitor("Counter: %d", value);
    repeat(1024) begin
      #1 enable_out = !enable_out;
    end
    $finish;
  end
 
endmodule

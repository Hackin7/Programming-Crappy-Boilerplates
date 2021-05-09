/**
 * 4 bit program counter - PC
 */
module program_counter
  (
   input        i_debug,
   // Count is reset to 0000 when i_reset goes high
   input        i_reset,
   input        i_increment,
   // Module output is tri-state; only enabled when i_enable_out is high
   input        i_enable_out,
   input        i_jump,
   // 4 bit count output to 8 bit bus
   inout [7:0] o_count
   );

   // Internal count register
   reg [7:0]    count = 0;
   reg [7:0]    count_buffer = 8'bzzzzzzzz;    
   assign o_count = count_buffer;
   
   always @(posedge i_increment) begin
      count <= count + 1; // It automatically overflows
      if(i_debug) $display("DEBUG: PC increment: %b",count);
   end

   always @(i_enable_out) begin
      if(i_enable_out) begin
         count_buffer <= count;
         if(i_debug) $display("DEBUG: PC write to bus: %b", {4'b0000, count});
      end else begin
         count_buffer <= 8'bzzzzzzzz;
      end
   end
   
   always @(posedge i_reset) begin
      count <= 0;
      if(i_debug) $display("DEBUG: PC reset: %b",count);
   end

  always @(posedge i_jump) begin
    count <= o_count;
  end
 
endmodule // program_counter

/**
 * SAP-1 whole system integration
 */
`include "program_counter/program_counter.v"
//`include "memory_address_register/memory_address_register.v"
//`include "mux_2x4/mux_2x4.v"
`include "ram_16x8/ram_16x8.v"
//`include "instruction_register/instruction_register.v"
`include "register/register.v"
`include "alu/alu.v"
`include "controller/controller.v"

module test;

   // Control-Sequencer output signals
   wire         ctl_halt;
   wire         ctl_memory_address_in;
   wire         ctl_ram_in;
   wire         ctl_ram_out;
   wire         ctl_instruction_out;
   wire         ctl_instruction_in;
   wire         ctl_register_A_in;
   wire         ctl_register_A_out;
   wire         ctl_alu_out;
   wire         ctl_alu_subtract;
   wire         ctl_register_B_in;
   wire         ctl_register_output_in;
   wire         ctl_program_counter_increment;
   wire         ctl_program_counter_out;
   wire         ctl_program_counter_jump;

   // Component wires
   wire [7:0]   bus; // Main system bus

   wire [3:0]   opcode; //Connects from IR to Controller
   
   // RAM should respond to control signals and program mode:   
   program_counter pc
     (
      .i_debug(1'b1),
      .i_reset(1'b0),
      .i_increment(ctl_program_counter_increment),
      .i_enable_out(ctl_program_counter_out),
      .o_count(bus)
      );
  // RAM ///////////////////////////////////////////////////
  wire ram_write;
  wire [7:0] ram_address;
  wire [15:0] ram_bus;
  reg [15:0] ram_i_program_data;
  ram_16x8 ram
     (
      .i_debug(1'b1), //i_debug_ram),
      .i_program_mode(1'b1), //i_program_mode),
      .i_program_data(ram_i_program_data),
      .i_address(ram_address),
      .i_write_enable(ram_write),
      .i_read_enable(ctl_ram_out),
      .io_data(ram_bus)
      ); 
  
  reg ram_i_write = 0; buf(ram_write, ram_i_write);
  reg [7:0] ram_i_address; assign ram_address = ram_i_address; //buf(ram_address, ram_i_address);
  reg ram_i_read =0; buf(ctl_ram_out, ram_i_read);
  // ALU Setup ////////////////////////////////////////////////////////////
   wire [7:0]   alu_A_in; //Connects from Register A to ALU
   wire [7:0]   alu_B_in; //Connects from Register B to ALU
   wire         alu_flag_zero; //Zero result flag from ALU to Controller
   wire         alu_flag_overflow; //Overflow result flag from ALU to Controller
    register register_A
     (
      .i_debug(1'b1),
      .i_reset(1'b0),
      .i_load_data(ctl_register_A_in),
      .i_send_data(ctl_register_A_out),
      .i_bus(bus),
      .o_bus(bus),
      .o_unbuffered(alu_A_in)
      );

   register register_B
     (
      .i_debug(1'b1),
      .i_reset(1'b0),
      .i_load_data(ctl_register_B_in),
      .i_bus(bus),
      .o_bus(), // Register B only outputs unbuffered, to the ALU.
      .o_unbuffered(alu_B_in)
      );

   alu alu
     (
      .i_a(alu_A_in),
      .i_b(alu_B_in),
      .i_subtract(ctl_alu_subtract),
      .i_send_result(ctl_alu_out),
      .o_flag_overflow(alu_flag_overflow),
      .o_flag_zero(alu_flag_zero),
      .o_bus(bus)
      );

  // Microcode Testing ////////////////////////////
  
  /////////////////////////////////////////////////
  initial begin
    $display("Start");
    $monitor("RAM Bus %b, %b", ram_bus, ram_i_write);
    // Loading into RAM
    ram_i_address = 0;
    ram_i_program_data = 16'b0000000011111111;
    ram_i_write = 1;
    #10 ram_i_write=0;
    #10 ram_i_read = 1;
    #10 ram_i_read=0;
    ram_i_address = 1;
    ram_i_program_data = 16'b1100100011111111;
    ram_i_write = 1;
    #10 ram_i_write=0;
    #10 ram_i_read = 1;
    $finish;
  end  
endmodule


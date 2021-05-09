/**
 * SAP-1 whole system integration
 */
`include "program_counter.v"
`include "ram.v"
`include "register.v"
`include "alu.v"
`include "controller.v"

module test(
    //input clk,
    output [7:0] bus
    );
   reg i_reset = 0;
   // Control-Sequencer output signals
   wire         ctl_halt;
   wire         ctl_memory_address_in;
   wire         ctl_ram_in;
   wire         ctl_ram_out;
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
   //wire [7:0]   bus; // Main system bus

   wire [7:0]   opcode; //Connects from IR to Controller

   // RAM should respond to control signals and program mode:
   program_counter pc
     (
      .i_debug(1'b1),
      .i_reset(i_reset),
      .i_increment(ctl_program_counter_increment),
      .i_enable_out(ctl_program_counter_out),
      .i_jump(ctl_program_counter_jump),
      .o_count(bus)
      );
  // RAM ///////////////////////////////////////////////////
  wire ram_write;
  wire [7:0] ram_address;
  wire [15:0] ram_bus;
  reg [15:0] ram_i_program_data;

  register mar //memory_address_register
     (
      .i_debug(1'b1), //i_debug_mar),
      .i_reset(i_reset),
      .i_load_data(ctl_memory_address_in),
      .i_send_data(),
      .i_bus(bus),
      .o_bus(),
      .o_unbuffered(ram_address)
      );

 given_ram ram
     (
      .i_debug(1'b1), //i_debug_ram),
      .i_program_mode(1'b1), //i_program_mode)
      .i_program_address(ram_i_address),
      .i_program_data(ram_i_program_data),
      .i_address(ram_address),
      .i_write_enable(ram_write),
      .i_read_enable(ctl_ram_out),
      .io_data(ram_bus),
      .o_data(bus)
      );

  reg ram_i_write = 0; buf(ram_write, ram_i_write);
  reg [7:0] ram_i_address; // assign ram_address = ram_i_address; //buf(ram_address, ram_i_address);

  //reg ram_i_read =0; buf(ctl_ram_out, ram_i_read);
  //assign opcode = ram_bus[15:8];
  //assign bus = (bus) ? bus : ram_bus[7:0];

  // ALU Setup ////////////////////////////////////////////////////////////
   wire [7:0]   alu_A_in; //Connects from Register A to ALU
   wire [7:0]   alu_B_in; //Connects from Register B to ALU
   wire         alu_flag_zero; //Zero result flag from ALU to Controller
   wire         alu_flag_overflow; //Overflow result flag from ALU to Controller
    register register_A
     (
      .i_debug(1'b1),
      .i_reset(i_reset),
      .i_load_data(ctl_register_A_in),
      .i_send_data(ctl_register_A_out),
      .i_bus(bus),
      .o_bus(bus),
      .o_unbuffered(alu_A_in)
      );

   register register_B
     (
      .i_debug(1'b1),
      .i_reset(i_reset),
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
  //reg clk = 0;
  controller control
     (
      .i_debug(1'b1), //i_debug_control),
      .i_clock(clk), //i_clock),
      .i_reset(i_reset),
      .i_instruction(ram_bus),
      .i_flag_overflow(alu_flag_overflow),
      .i_flag_zero(alu_flag_zero),
      .o_halt(ctl_halt),
      .o_memory_address_in(ctl_memory_address_in),
      .o_ram_in(ctl_ram_in),
      .o_ram_out(ctl_ram_out),
      // .o_instruction_in(ctl_instruction_in),
      // .o_instruction_out(ctl_instruction_out),
      .o_register_a_in(ctl_register_A_in),
      .o_register_a_out(ctl_register_A_out),
      .o_alu_out(ctl_alu_out),
      .o_alu_subtract(ctl_alu_subtract),
      .o_register_b_in(ctl_register_B_in),
      .o_register_output_in(ctl_register_output_in),
      .o_program_counter_increment(ctl_program_counter_increment),
      .o_program_counter_out(ctl_program_counter_out),
      .o_program_counter_jump(ctl_program_counter_jump)
      );

  reg clk=0;
  // Testing Code ////////////////////////////////////
  initial begin
    //////////////////////////////////////////////////
    $display("### Load RAM #################################");
    $monitor("############### RAM Bus %b, Write %b, Normal bus %b #############", ram_bus, ram_i_write, bus);
    $monitor("Overflow %b %b",alu_flag_overflow, ctl_alu_out);
    // Loading into RAM
    ram_i_address = 0;
    ram_i_program_data = {8'd0, 8'd255};
    ram_i_write = 1;
    $display("Writing");
    #10 ram_i_write=0;
    //#10 ram_i_read = 1;
    #10
    //ram_i_read=0;
    ram_i_address = 1;
    ram_i_program_data = {8'd1, 8'd127}; // Max positive value
    ram_i_write = 1;
    #10 ram_i_write=0;
    #10
    ram_i_address = 2;
    ram_i_program_data = {8'd2, 8'd1};
    //ram_i_program_data = {8'd3, 8'd127};
    ram_i_write = 1;
    #10 ram_i_write=0;
    #10
    //ram_i_read=0;
    ram_i_address = 3;
    ram_i_program_data = {8'd7, 8'd2}; // 6 JMP, 7 JC  8 JZ
    ram_i_write = 1;
    #10 ram_i_write=0;
    // Microcode Test ////////////////////////////////

    #10
    $display("");
    $display("### Microcode Test ##########################$#####");
    #10 clk = 1;
    #10 clk = 0;
    $display("### Bus: %0b ###############", bus);
    #10 clk = 1;
    #10 clk = 0;
    $display("### Bus: %0b ###############", bus);

    //repeat(2) begin
    //  #10 clk = 1;
    //  #10 clk = 0;
      //$display("### Bus: %0b ###############", bus);
    //end
    //$finish;
  end
endmodule

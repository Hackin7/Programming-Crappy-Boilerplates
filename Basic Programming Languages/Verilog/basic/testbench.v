/*
Compilation Steps
  iverilog -o hello hello.v
  vvp hello
*/


/////////////////////////////////////////////////////////////////////////
module data_sample;

  wire      wire_a;             // 0, 1, x(unknown), z(high impedence)
  reg       reg_a;
    
  integer  	int_a; 				// Integer variable
  real 		real_b; 			// Real variable
  time 		time_c; 
  
  reg [8*11:1] str1;            //Register Vector
  reg [8*5:1]  str2;
  reg [8*20:1] str3;
  
  // msb:lsb
  reg [7:0] reg_vector;    
  reg       reg_array[7:0];    // Register Array 
  
  initial begin
    int_a 	= 32'hcafe_1234; 	// Assign an integer value
    real_b 	= 0.1234567; 		// Assign a floating point value
    #1; 						// Advance simulation time by 20 units
    time_c 	= $time; 			// Assign current simulation time
    
    str1 = "Hello World";
    str2 = "Hello World";
    str3 = "Hello World";
    
    //Choose
    reg_vector = 11;//8'b10001010';
    reg_vector = 8'b10001010; //8'bzzzzzzxzz;
    reg_vector = 8'h41;       //Hexdata
    reg_vector [7:4] = 4'h5; //Slicing
    reg_vector[0] = 0;
    
    reg_array[0] = 1;
    reg_array[7] = 0;
    
    // Now print all variables using $display system task
    $display ("### Electronics ################");
    $display ("wire_a 	= 0x%0h", wire_a); //High impedence
    $display ("reg_a 	= 0x%0h", reg_a); //Undefined
    reg_a = 1;
    $display ("reg_a 	= 0x%0h", reg_a); //Undefined
    
    $display ("### Numbers ################");
    $display ("int_a 	= 0x%0h", int_a);
    $display ("real_b 	= %0.5f", real_b);
    $display ("time_c 	= %0t", time_c);
    
    $display ("### Strings ################");
    $display ("str1 = %s", str1);
    $display ("str2 = %s", str2);
    $display ("str3 = %s", str3);
    
    $display ("### Vectors and Arrays ################");
    $display ("reg_vector   = 0x%0h, %0d", reg_vector, reg_vector);
    $display ("reg_array[0] = 0x%0h", reg_array[0]);
    $display ("reg_array[7] = 0x%0h", reg_array[7]);
    //$display ("reg_array[7] = 0x%0h", reg_array[8]); //This code should fail
    
  end
    
endmodule 

/////////////////////////////////////////////////////////////////////////
// https://www.chipverify.com/verilog/verilog-always-block
module initial_always;
    reg start = 0;    
    
    initial begin
       #10 start = 1;
       #10 start = 0;
    end
    
    always #10 $display("%0t done", $time); // Loop every 10 time units
    
    always @ (start) begin
       $display("Start");
    end
    always @ (!start) begin
       $display("Start NOT");
    end
    always @ (posedge start) begin
       $display("Start Posedge"); 
    end
    always @ (negedge start) begin
       $display("Start Negedge"); 
    end
endmodule

/////////////////////////////////////////////////////////////////////////
// https://www.chipverify.com/verilog/verilog-control-block
module control_structures;
    integer i=5; 
    initial begin
        // Selection ///////////////////
        if (1) begin
          $display("Ran 1");
        end else if (1) begin
          $display("Ran 2");
        end else begin
          $display("Ran 3");
        end
        
        case (2)
          1: $display("Case 1");
          2: begin $display("Case 2"); end
          default: $display("default"); 
        endcase
        // Loops //////////////////////
        repeat(4) begin
			$display("This is a new iteration ...");
		end
		
		while (i > 0) begin
			$display ("While loop Iteration #%0d", i);
			i = i - 1;
		end
        for(i=0;i<10;i=i+1) begin
        	$display ("For loop Iteration #%0d", i);
		end
    end
endmodule

/////////////////////////////////////////////////////////////////////////
module test;
  //// Variables and Data Types /////////
  
  data_sample d();
  initial_always i();
  control_structures c();
  // Initial
  initial begin
    $display("Hello, World");
    
    # 100 // Wait 100 time units
    
    $finish ; // end simualtion
    
  end
  
    
endmodule

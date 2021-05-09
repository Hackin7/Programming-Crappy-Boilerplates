/*
Compilation Steps
  iverilog -o hello hello.v
  vvp hello
*/

module test;
  reg a, b;
  wire c, d, e, f, g, h, i, j, k, l, m, n, o, p;

  gates gates_test(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);

  integer count;

  initial begin
    {a, b} = 0;
    $monitor("[T=%0t a=%0b b=%0b : c(and)=%0b d(or)=%0b e(xor)=%0b f(nand)=%0b g(nor)=%0b h(xnor)=%0b i(buf)=%0b j(not)=%0b] k(bufif0)=%0b l(notif0)=%0b m(bufif1)=%0b n(notif1)=%0b o(pullup)=%0b p(pulldown)=%0b",
             $time, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p); 
   
    for (count=0;count<10;count=count+1) begin
    #1 a <= $random;
       b <= $random;
    end

  end
endmodule

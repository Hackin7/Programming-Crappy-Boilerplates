module sr_flip_flop(Q, Qn, S, R);
   output Q;
   output Qn;
   input  S;
   input  R;

   nor(Qn, S, Q);
   nor(Q, R, Qn);
endmodule 

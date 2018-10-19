option limrow = 0;
option limcol = 0;
option sysout = off;
option solprint = off;






$include 1_CGEP2P.gms
*********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

*********************OPTIMIZATION RUNS********************************

***********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

file fx1 /newfile.txt/;
fx1.ap = 1;

Parameter tax2 /0.068/;
Set count /1*100/;


Loop(count,
*Fixing the Final Demand
P2Pf.FX('7') = 4;
tauz('PR1') = tax2;
tax2 = tax2 + 0.005;



****************************Solving forCGE local+ global + VC+ Equipment Scale emission**********************************************


Solve CGEP2P maximizing Profit  using NLP;
*Solve CGEP2P minimizing Ez3 using NLP;

Display ANSWER;
Display ANSWER;
Display ANSWER;
Display ANSWER;
Display ANSWER;
Display ANSWER;




put fx;
fx.nd = 4;
put p2pf.L('7')                   ;
put tauz('PR1')                   ;
put Ez1.L                         ;
put Ez2.L                         ;
put Rate.L                        ;
put Efficiency.L                  ;
put Cost.L                        ;
put size_of_conventional_flow.L   ;
put size_of_emergent_flow.L       ;
put F2.L                          ;
put F1.L
put c1.L;
put c2.L; 
put c3.L;
put p_F1_1.L                      ;
put p_F1_2.L                      ;
put p_F2;
putclose;



Display Value_chain_emission.L,Equipment_scale_emission.L,env_l.L,env_g.L;
Display Ez1.L,EZ2.L,Ez3.L,Ez4.L;

Display F1.L,F2.L,POut.L,Rate.L,Efficiency.L,EqCO2.L,Cost.L,size_of_conventional_flow.L,size_of_emergent_flow.L;


Display p_pOut,Profit.L;

Display taud, tauz, taum;


Display P2Px.L,P2Pint.L,P2Ps.L,F1.L,size_of_conventional_flow.L,size_of_emergent_flow.L,demand_to_PR1.L,demand_to_PR2.L;
Display UU.L, Y.L, F.L, Z.L, D.L, AQ.L,Xp.L,IM.L,Xv.L,pf.L, pz.L,Xg.L,E.L,
        pq.L, pe.L, pm.L;
);




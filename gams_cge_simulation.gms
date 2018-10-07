





*option limrow = 0;
*option limcol = 0;
*option sysout = off;
*option solprint = off;


$include 3_CGE_P2P.gms
*********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

*********************OPTIMIZATION RUNS********************************

***********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************


Parameter demand /5/;
Set count /1*1/;


Loop(count,
*Fixing the Final Demand
P2Pf.LO('5') = demand;
demand = demand + 50;




****************************Solving forCGE local+ global + VC+ Equipment Scale emission**********************************************



Solve CGEP2P minimizing Ez4 using NLP;


Display ANSWER;
Display ANSWER;
Display ANSWER;
Display ANSWER;
Display ANSWER;
Display ANSWER;

put fx;
put p2pf.L('5')                   ;
put '    '                        ;
put Ez1.L                         ;
put '    '                        ;
put Ez2.L                         ;
put '    '                        ;
put Ez3.L                         ;
put '    '                        ;
put Ez4.L                         ;
put '    '                        ;
put Rate.L                        ;
put '    '                        ;
put Efficiency.L                  ;
put '    '                        ;
put Cost.L                        ;
put '    '                        ;
put size_of_conventional_flow.L   ;
put '    '                        ;
put size_of_emergent_flow.L       ;
put '    '                        ;
put p_F1_1.L                      ;
put '    '                        ;
put p_F1_2.L                      ;
put '    '                        ;
put demand_to_PR1.L            ;
put '    '                        ;
put demand_to_PR2.L            ;
putclose;



Display Value_chain_emission.L,Equipment_scale_emission.L,env_l.L,env_g.L;
Display Ez1.L,EZ2.L,Ez3.L,Ez4.L;

Display F1.L,F2.L,POut.L,Rate.L,Efficiency.L,EqCO2.L,Cost.L;

Display P2Px.L,P2Pint.L,P2Ps.L,F1.L,size_of_conventional_flow.L,size_of_emergent_flow.L,demand_to_PR1.L,demand_to_PR2.L;
Display UU.L, Y.L, F.L, Z.L, D.L, AQ.L,Xp.L,IM.L,Xv.L,pf.L, pz.L,Xg.L,E.L,
        pq.L, pe.L, pm.L;
);

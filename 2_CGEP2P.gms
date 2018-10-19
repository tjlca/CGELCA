$inlinecom /* */
$offlisting
$offsymxref
$offsymlist
$offinclude

option limrow = 0;
option limcol = 0;
option sysout = off;
option solprint = off;


$ontext

Title: Supporting Information 1 for "xxx"

Author:  Tapajyoti Ghosh, Kyuha Lee  and Bhavik R. Bakshi*
*Corresponding author: bakshi.2@osu.edu


$title A Standard CGE Model in Ch. 6 (STDCGE,SEQ=276)

$onText
No description.


Hosoe, N, Gasawa, K, and Hashimoto, H
Handbook of Computible General Equilibrium Modeling
University of Tokyo Press, Tokyo, Japan, 2004

Keywords: nonlinear programming, general equilibrium model, social accounting
          matrix, utility maximization problem
$offText

Set
   u    'SAM entry' / PR1, PR2, CAP, LAB, IDT, TRF, HOH, GOV, INV, EXT /
   i(u) 'goods'     / PR1, PR2                                         /
   h(u) 'factor'    /           CAP, LAB                               /;

Alias (u,v), (i,j), (h,r);

Table SAM(u,v) 'social accounting matrix'
         PR1   PR2   CAP   LAB   IDT   TRF   HOH   GOV   INV   EXT
   PR1    21     8                            20    19    16     8
   PR2    17     9                            30    14    15     4
   CAP    20    30
   LAB    15    25
   IDT     5     4
   TRF     1     2
   HOH                50    40
   GOV                             9     3    23
   INV                                        17     2          12
   EXT    13    11                                                ;

* Loading the initial values
Parameter
   Y0(j)   'composite factor'
   F0(h,j) 'the h-th factor input by the j-th firm'
   X0(i,j) 'intermediate input'
   Z0(j)   'output of the j-th good'
   Xp0(i)  'household consumption of the i-th good '
   Xg0(i)  'government consumption'
   Xv0(i)  'investment demand'
   E0(i)   'exports'
   M0(i)   'imports'
   Q0(i)   "Armington's composite good"
   D0(i)   'domestic good'
   Sp0     'private saving'
   Sg0     'government saving'
   Td0     'direct tax'
   Tz0(j)  'production tax'
   Tm0(j)  'import tariff'
   FF(h)   'factor endowment of the h-th factor'
   Sf      'foreign saving in US dollars'
   pWe(i)  'export price in US dollars'
   pWm(i)  'import price in US dollars'
   tauz(i) 'production tax rate'
   taum(i) 'import tariff rate';

Td0     = SAM("GOV","HOH");
Tz0(j)  = SAM("IDT",j);
Tm0(j)  = SAM("TRF",J);
F0(h,j) = SAM(h,j);
Y0(j)   = sum(h, F0(h,j));
X0(i,j) = SAM(i,j);
Z0(j)   = Y0(j) +sum(i, X0(i,j));
M0(i)   = SAM("EXT",i);
tauz(j) = Tz0(j)/Z0(j);
taum(j) = Tm0(j)/M0(j);
Xp0(i)  = SAM(i,"HOH");
FF(h)   = SAM("HOH",h);
Xg0(i)  = SAM(i,"GOV");
Xv0(i)  = SAM(i,"INV");
E0(i)   = SAM(i,"EXT");
Q0(i)   = Xp0(i)+Xg0(i)+Xv0(i)+sum(j, X0(i,j));
D0(i)   = (1+tauz(i))*Z0(i)-E0(i);
Sp0     = SAM("INV","HOH");
Sg0     = SAM("INV","GOV");
Sf      = SAM("INV","EXT");
pWe(i)  = 1;
pWm(i)  = 1;


*display Y0, F0, X0, Z0, Xp0, Xg0, Xv0, E0, M0, Q0, D0, Sp0, Sg0, Td0, Tz0, Tm0
*        FF, Sf, tauz, taum;

* Calibration
Parameter
   sigma(i) 'elasticity of substitution'
   psi(i)   'elasticity of transformation'
   eta(i)   'substitution elasticity parameter'
   phi(i)   'transformation elasticity parameter';



sigma(i) = 2;
psi(i)   = 2;
eta(i)   = (sigma(i) - 1)/sigma(i);
phi(i)   = (psi(i) + 1)/psi(i);

Parameter
   alpha(i)  'share parameter in utility func.'
   beta(h,j) 'share parameter in production func.'
   b(j)      'scale parameter in production func.'
   ax(i,j)   'intermediate input requirement coeff.'
   ay(j)     'composite fact. input req. coeff.'
   mu(i)     'government consumption share'
   lambda(i) 'investment demand share'
   deltam(i) 'share par. in Armington func.'
   deltad(i) 'share par. in Armington func.'
   gamma(i)  'scale par. in Armington func.'
   xid(i)    'share par. in transformation func.'
   xie(i)    'share par. in transformation func.'
   theta(i)  'scale par. in transformation func.'
   ssp       'average propensity for private saving'
   ssg       'average propensity for gov. saving'
   taud      'direct tax rate';

alpha(i)  =  Xp0(i)/sum(j, Xp0(j));
beta(h,j) =  F0(h,j)/sum(r, F0(r,j));
b(j)      =  Y0(j)/prod(h, F0(h,j)**beta(h,j));
ax(i,j)   =  X0(i,j)/Z0(j);
ay(j)     =  Y0(j)/Z0(j);
mu(i)     =  Xg0(i)/sum(j, Xg0(j));
lambda(i) =  Xv0(i)/(Sp0+Sg0+Sf);
deltam(i) = (1+taum(i))*M0(i)**(1-eta(i))/((1+taum(i))*M0(i)**(1-eta(i)) + D0(i)**(1-eta(i)));
deltad(i) =  D0(i)**(1-eta(i))/((1+taum(i))*M0(i)**(1-eta(i)) + D0(i)**(1-eta(i)));
gamma(i)  =  Q0(i)/(deltam(i)*M0(i)**eta(i)+deltad(i)*D0(i)**eta(i))**(1/eta(i));
xie(i)    =  E0(i)**(1-phi(i))/(E0(i)**(1-phi(i))+D0(i)**(1-phi(i)));
xid(i)    =  D0(i)**(1-phi(i))/(E0(i)**(1-phi(i))+D0(i)**(1-phi(i)));
theta(i)  =  Z0(i)/(xie(i)*E0(i)**phi(i)+xid(i)*D0(i)**phi(i))**(1/phi(i));
ssp       =  Sp0/sum(h, FF(h));
ssg       =  Sg0/(Td0+sum(j, Tz0(j))+sum(j, Tm0(j)));
taud      =  Td0/sum(h, FF(h));

*display alpha, beta,  b,   ax,  ay, mu, lambda, deltam, deltad, gamma, xie
*        xid,   theta, ssp, ssg, taud;

Variable
   Y(j)    'composite factor'
   F(h,j)  'the h-th factor input by the j-th firm'
   X(i,j)  'intermediate input'
   Z(j)    'output of the j-th good'
   Xp(i)   'household consumption of the i-th good'
   Xg(i)   'government consumption'
   Xv(i)   'investment demand'
   E(i)    'exports'
   IM(i)    'imports'
   AQ(i)    "Armington's composite good"
   D(i)    'domestic good'
   pf(h)   'the h-th factor price'
   py(j)   'composite factor price'
   pz(j)   'supply price of the i-th good'
   pq(i)   "Armington's composite good price"
   pe(i)   'export price in local currency'
   pm(i)   'import price in local currency'
   pd(i)   'the i-th domestic good price'
   epsilon 'exchange rate'
   Sp      'private saving'
   Sg      'government saving'
   Td      'direct tax'
   Tz(j)   'production tax'
   Tm(i)   'import tariff'
   UU      'utility [fictitious]';



Equation
   eqpy(j)   'composite factor agg. func.'
   eqF(h,j)  'factor demand function'
   eqX(i,j)  'intermediate demand function'
   eqY(j)    'composite factor demand function'
   eqpzs(j)  'unit cost function'
   eqTd      'direct tax revenue function'
   eqTz(j)   'production tax revenue function'
   eqTm(i)   'import tariff revenue function'
   eqXg(i)   'government demand function'
   eqXv(i)   'investment demand function'
   eqSp      'private saving function'
   eqSg      'government saving function'
   eqXp_1    'household demand function'
   eqXp_2    'household demand function' 
*   eqXp(i)    'household demand function'
   eqpe(i)   'world export price equation'
   eqpm(i)   'world import price equation'
   eqepsilon 'balance of payments'
   eqpqs(i)  'Armington function'
   eqM(i)    'import demand function'
   eqD(i)    'domestic good demand function'
   eqpzd(i)  'transformation function'
   eqDs(i)   'domestic good supply function'
   eqE(i)    'export supply function'
   eqpqd(i)  'market clearing cond. for comp. good'
   eq_pf(h)   'factor market clearing condition'
   obj       'utility function [fictitious]';

* domestic production
eqpy(j)..   Y(j)   =e= b(j)*prod(h, F(h,j)**beta(h,j));

eqF(h,j)..  F(h,j) =e= beta(h,j)*py(j)*Y(j)/pf(h);

eqX(i,j)..  X(i,j) =e= ax(i,j)*Z(j);

eqY(j)..    Y(j)   =e= ay(j)*Z(j);

eqpzs(j)..  pz(j)  =e= ay(j)*py(j) + sum(i, ax(i,j)*pq(i));

* government behavior
eqTd..      Td     =e= taud*sum(h, pf(h)*FF(h));

eqTz(j)..   Tz(j)  =e= tauz(j)*pz(j)*Z(j);

eqTm(i)..   Tm(i)  =e= taum(i)*pm(i)*IM(i);

eqXg(i)..   Xg(i)  =e= mu(i)*(Td + sum(j, Tz(j)) + sum(j, Tm(j)) - Sg)/pq(i);

* investment behavior
eqXv(i)..   Xv(i)  =e= lambda(i)*(Sp + Sg + epsilon*Sf)/pq(i);

* savings
eqSp..      Sp     =e= ssp*sum(h, pf(h)*FF(h));

eqSg..      Sg     =e= ssg*(Td + sum(j, Tz(j)) + sum(j, Tm(j)));

* household consumption
*eqXp(i)..   Xp(i)  =e= alpha(i)*(sum(h, pf(h)*FF(h)) - Sp - Td)/pq(i);


*Creating the elasticity of substitution for utility
Parameter sigma_u Elasticity of substitution for consumption;

sigma_u = 1;

* Creating a new household consumption function using CES
*eqXp_1..   Xp('PR1') =e= 2*alpha('PR1')*(sum(h, pf(h)*FF(h)) - Sp -Td ) * (pq('PR1')**(-sigma_u)) /( (pq('PR1')**(1-sigma_u)) + ((alpha('PR1')/alpha('PR2'))**(1-sigma_u)) * (pq('PR2')**(1-sigma_u)) );
*eqXp_2..   Xp('PR2') =e= 2*alpha('PR2')*(sum(h, pf(h)*FF(h)) - Sp -Td ) * (pq('PR2')**(-sigma_u)) /( (pq('PR2')**(1-sigma_u)) + ((alpha('PR2')/alpha('PR1'))**(1-sigma_u)) * (pq('PR1')**(1-sigma_u)) );

* Creating a new household consumption function using CES
eqXp_1..   Xp('PR1') =e= 2*alpha('PR1')*(sum(h, pf(h)*FF(h)) - Sp -Td ) * (pq('PR1')**(-sigma_u)) /( (pq('PR1')**(1-sigma_u)) + (pq('PR2')**(1-sigma_u)) );
eqXp_2..   Xp('PR2') =e= 2*alpha('PR2')*(sum(h, pf(h)*FF(h)) - Sp -Td ) * (pq('PR2')**(-sigma_u)) /( (pq('PR2')**(1-sigma_u)) + (pq('PR1')**(1-sigma_u)) );






* international trade
eqpe(i)..   pe(i)  =e= epsilon*pWe(i);

eqpm(i)..   pm(i)  =e= epsilon*pWm(i);

eqepsilon.. sum(i, pWe(i)*E(i)) + Sf =e= sum(i, pWm(i)*IM(i));

* Armington function
eqpqs(i)..  AQ(i)   =e=  gamma(i)*(deltam(i)*IM(i)**eta(i) + deltad(i)*D(i)**eta(i))**(1/eta(i));

eqM(i)..    IM(i)   =e= (gamma(i)**eta(i)*deltam(i)*pq(i)/((1+taum(i))*pm(i)))**(1/(1-eta(i)))*AQ(i);

eqD(i)..    D(i)   =e= (gamma(i)**eta(i)*deltad(i)*pq(i)/pd(i))**(1/(1-eta(i)))*AQ(i);

* transformation function
eqpzd(i)..  Z(i)   =e=  theta(i)*(xie(i)*E(i)**phi(i)+xid(i)*D(i)**phi(i))**(1/phi(i));

eqE(i)..    E(i)   =e= (theta(i)**phi(i)*xie(i)*(1+tauz(i))*pz(i)/pe(i))**(1/(1-phi(i)))*Z(i);

eqDs(i)..   D(i)   =e= (theta(i)**phi(i)*xid(i)*(1+tauz(i))*pz(i)/pd(i))**(1/(1-phi(i)))*Z(i);

* market clearing condition
eqpqd(i)..  AQ(i)   =e= Xp(i) + Xg(i) + Xv(i) + sum(j, X(i,j));

eq_pf(h)..   sum(j, F(h,j)) =e= FF(h);

* fictitious objective function
obj..       UU     =e= prod(i, Xp(i)**alpha(i));

* Initializing variables
Y.l(j)    = Y0(j);
F.l(h,j)  = F0(h,j);
X.l(i,j)  = X0(i,j);
Z.l(j)    = Z0(j);
Xp.l(i)   = Xp0(i);
Xg.l(i)   = Xg0(i);
Xv.l(i)   = Xv0(i);
E.l(i)    = E0(i);
IM.l(i)    = M0(i);
AQ.l(i)    = Q0(i);
D.l(i)    = D0(i);
pf.l(h)   = 1;
py.l(j)   = 1;
pz.l(j)   = 1;
pq.l(i)   = 1;
pe.l(i)   = 1;
pm.l(i)   = 1;
pd.l(i)   = 1;
epsilon.l = 1;
Sp.l      = Sp0;
Sg.l      = Sg0;
Td.l      = Td0;
Tz.l(j)   = Tz0(j);
Tm.l(i)   = Tm0(i);

* Setting lower bounds to avoid division by zero
Y.lo(j)    = 0.00001;
F.lo(h,j)  = 0.00001;
X.lo(i,j)  = 0.00001;
Z.lo(j)    = 0.00001;
Xp.lo(i)   = 0.00001;
Xg.lo(i)   = 0.00001;
Xv.lo(i)   = 0.00001;
E.lo(i)    = 0.00001;
IM.lo(i)    = 0.00001;
AQ.lo(i)    = 0.00001;
D.lo(i)    = 0.00001;
pf.lo(h)   = 0.00001;
py.lo(j)   = 0.00001;
pz.lo(j)   = 0.00001;
pq.lo(i)   = 0.00001;
pe.lo(i)   = 0.00001;
pm.lo(i)   = 0.00001;
pd.lo(i)   = 0.00001;
epsilon.lo = 0.00001;
Sp.lo      = 0.00001;
Sg.lo      = 0.00001;
Td.lo      = 0.00001;
Tz.lo(j)   = 0.0000;
Tm.lo(i)   = 0.0000;

* numeraire
pf.fx("LAB") = 1;







*******************************************************************************



********************************************************************************************************************************************************************
****It has two prices because instead of having two competiting industries producing two products, we are assuming that the product is the same. This makes the model Rectangualar. Its just an assumption to help us with easier calculations. No methological signifance.
*price of the product that is produced by two sectors. The product even though it is one has two prices.
********************************************************************************************************************************************************************




Positive Variable
p_F1_1    Actual Price of VC scale product F1(Conventional($|kg)
p_F1_2    Actual Price of VC Scale product F1(Clean)($|kg);


Parameter
p0R1   Actual price at initial state  /2/
p0R2   Actual price at initial state  /2/;



*******************************************************************************************

*******************LInkign the price to the CGE model****************************************

********************************************************************************************
***********************************************************************************************
Parameter pq0(i);


pq0('PR1') = 1;
pq0('PR2') = 1;

*We had  an actual price at the initial state. That price is changed to the numeraire in a CGE model. We know that ratio. We use that ratio to convert the prices from the CGE model to actual prices of the products. THis has the assumption that the ratio of conversiom from the actual price to the relative prices in the CGE model never changes from equilibrium to equilibrium.

*2.8 , 1.5 etc. are just constants to determine a price.

Equation Link1, Link2;
Link1.. p_F1_1 =E= (pq('PR1') * 2.1 + pq('PR1') * 1.1) * p0R1/pq0('PR1') ;
Link2.. p_F1_2 =E= (pq('PR2') * 2.6 + pq('PR2') * 1.2) * p0R2/pq0('PR2');

*We are assuming that price is higher for emergent technology



*A very important property is that in normal IOLCA models we have an environmental impact factors in $/kg. These factors are determined by the total emission divided by the total economic throughout. But if price changes and flow doesnt change then then the economic throughput will also change.

*This results in a problem. If price increases IO model will overestimate emissions for the same production. If price decreases, IO model will underestimate emissions. Thus we need to change the impact factor according to the prices.

**********************************************************

*Storing before shock environmental impact in a parameter

*********************************************************

*Here we are multiplying with the relative production quantity. Thus we need to change to actual value by multiplying with relative price. That relative price is 1. Or pq0('PR1')



*Environmental emission are
*0.9kg/$ for PR1
*2kg/$ for PR2



parameter env0_l environmental Impact
          env0_g global env impact
          UU0  utility;
*env0_l = 73 * 0.9 * pq0('PR1')/p0R1  + 72 * 2 * pq0('PR2')/p0R2;
*env0_g = 84 * 0.9 * pq0('PR1')/p0R1  + 85 * 2 * pq0('PR2')/p0R2;


*The mistake was that I changed the quantity to actual quantity. But thats irrevelant because we are assuming we dont know the environmental impact / physcial quantity. We need the economic value.\\

*So we change the equation.
env0_l = 73 * 0.9 * pq0('PR1')  + 72 * 2 * pq0('PR2');
env0_g = 84 * 0.9 * pq0('PR1')  + 85 * 2 * pq0('PR2');



display env0_l;
display env0_g;


******************************************************************************************

*Environmental Impact measurement

*****************************************************************************************


variable env_l local environmental impact
         env_g global environmental impact which includes imports;
equation impact1,impact2;

*Same changes. The Z is the relative quantity. Multiply that with relative price to get actual economic value.

*We have derive the emission factor / kg of product.So just need to get the actual quantity and multiply with the values.

*The assumption here is that the emission factor / kg of product does not change from equilibrium to equilbrium.

*These does not make sense. If we can have the actual product value we will just
*use life cycle inventories.

*The problem is that most times we dont know the price. So no idea of knowing the actual quantity.

*Its assumed that we know the actual price.

*Another assumption is that the ratio of NOrmalized price to actual price does not change with equilibrium.


*THe values are 1.8 kgCO2 / kg product1 and 5 kgCO2 / kg product2

*Correcting old equation
*impact1.. env_l =E=  Z('PR1') * 0.9 * pq0('PR1')/p0R1 *  pq0('PR1')/pq('PR1')+ Z('PR2') * 2 * pq0('PR2')/p0R2 * pq0('PR2')/pq('PR2')- env0_l ;
*impact2.. env_g =E=  AQ('PR1') * 0.9 * pq0('PR1')/p0R1 * pq0('PR1')/pq('PR1')+ AQ('PR2') * 2 * pq0('PR2')/p0R2 * pq0('PR2')/pq('PR2')- env0_g ;


*The prices need to be varied So I am using these equations to determine the per kg emissions rather than a fixed value.

Parameter derived_IF_PR1,derived_IF_PR2;
derived_IF_PR1 = 73*1*0.9/(73*1)*p0R1;
derived_IF_PR2 = 72*1*2/(72*1)*p0R2;


impact1.. env_l =E=  Z('PR1') * pq0('PR1')/p0R1 * derived_IF_PR1      +     Z('PR2') * derived_IF_PR2 * pq0('PR2')/p0R2 ;
impact2.. env_g =E= (Z('PR1') + IM('PR1')) * derived_IF_PR1 * pq0('PR1')/p0R1      +     (Z('PR2') + IM('PR2')) * derived_IF_PR2 * pq0('PR2')/p0R2;


**********WE HAVE CHANGED THE VALUE CHAIN SCALE EMISSION TO REPRESENT ATTRIBUTIONAL DATA**************************************



**********************************************************************************

***********************TECHNOLOGY CHOICE MODEL************************************

**********************************************************************************


*******************************************************************************

********* 1. Product/Commodity Transaction and Intervention Equations *********

*******************************************************************************

Sets
i_k       P2P multiscale processes and sectors     /1*8/
j_k        P2P multiscale products and commodities /1*7/
m(i_k)     Value chain scale processes             /3*7/
n(j_k)     Value chain scale products              /3*6/
o(i_k)     Equipment scale process                 /8/
p(j_k)     Equipment scale Product                 /7/
q          Environmental interventions             /1*2/
eu(i_k)     Economy_scale process                   /1*2/
ev(j_k)     Economy_scale product                   /1*2/

;

$ontext
*** Value chain scale processes ***
m(i_k) = 3: Conventional F1 manufacturing process (kg)
m(i_k) = 4: Emerging F1 manufacturing process (kg)
m(i_k) = 5: F2 manufacturing process (kg)
m(i_k) = 6: Electricity generation process (kWh)
m(i_k) = 7: Diesel-fueled truck transportation process (ton*km)

*** Value chain scale products ***
n(j_k) = 3: F1 product (kg)
n(j_k) = 4: F2 product (kg)
n(j_k) = 5: Electricity product (kWh)
n(j_k) = 6: Diesel-fueled truck transportation product (ton*km)

*** Equipment scale process ***
o(i_k) = 8: Equipment scale engineering process (kg)
p(j_k) = 7: P Output product (kg)

*** Equipment scale product ***
q = 1: CO2 emissions (kgCO2eq)
q = 2: NOx emissions (kgNeq)
$offtext

Alias(m,m1);


Positive Variables
P2Ps(i_k)  P2P multiscale scaling vector
P2Pf(j_k)  P2P multiscale final demand vector;
;

P2Pf.FX(j_k)$(ord(j_k) ne 7) = 0;




*** Price of VC and Eq scale products to convert physical unit to monetary unit
Scalars
p_F2    Price of VC scale product F2 ($|kg)                              /5.1/
p_Elec  Price of VC scale product electricity  ($|kWh)                   /0.1/
p_Truck Price of VC scale product truck transportation ($|(ton*km))      /0.1/
p_POut  Price of Eq scale product P                                      /40/
;




********************************************************************************************************************************************************************
****It has two prices because instead of having two competiting industries producing two products, we are assuming that the product is the same. This makes the model Rectangualar. Its just an assumption to help us with easier calculations. No methological signifance.
*price of the product that is produced by two sectors. The product even though it is one has two prices.
********************************************************************************************************************************************************************




************************* 1-1. Equipment Scale Model **************************

*******************************************************************************

Variables
F1       Product F1 input flow (kg per h)
F2       Product F2 input flow (kg per h)
POut     Product P output flow (kg per h)
Rate     Design variable at the equipment scale
EqCO2    Point source CO2 emissions at the equipment scale (kgCO2 per h)
Profit   Profit ($)
PV       Present value with an interst rate of 7% for 20 years ($)
TCI      Total capital investment ($)
NPV      Net present value ($)
tNPVn    Temporary variable to calculate normalized NPV
NPVn     Normalized NPV ($ per kg of product P)
NPVt     NPV for producing the final demand ($)
;

*Equipment scale model constraints (0.1 <= Rate({z}) <= 0.9)
Rate.LO = 0.1;
Rate.UP = 0.5;
Rate.L = 0.5;
Scalar Eqs Eq scale process scaling vector to convert h to y /8000/;



Variable Efficiency Defined for controlling how efficient the process is 0.1<====<0.9;

*Efficiency.LO = 0.5;
*Efficiency.UP = 1;
Efficiency.L = 0.5;

Equations Eng1,Eng2,Eng4,Eng5,Eng6;
Eng1.. F2 =e= F1*(Efficiency)**1.9;
Eng2.. POut =E= F1 + F2;
Eng4.. EqCO2 =e= (F1**5+ F2** 3)**0.5;





Variables
Eqmake(o,p)      Eq scale make matrix (Eq process by Eq product)
Eqint(q,o)       Eq scale intervention matrix (intervention by Eq process)
;

Equations Eq1,Eq2;

Eq1(o,p).. Eqmake(o,p) =e= POut;
Eq2(o).. Eqint('1',o) =e= EqCO2;

Eqint.FX('2',o) = 0;
* No NOx emissions from the equipment scale model.



*******************************************************************************

*************** 1-2. Value Chain-Equipment Upstream Cutoff Flow ***************

*******************************************************************************

Variable EqUpCut(n,o) Upstream cutoff flow from VC to Eq (VC product by Eq process);

Equation EqUpCut1,EqUpCut2;


*Remember to make negative in P2P
EqUpCut1(o).. EqUpCut('3',o) =e= F1;
EqUpCut2(o).. EqUpCut('4',o) =e= F2;

EqUpCut.FX('5',o) = 0;
EqUpCut.FX('6',o) = 0;




*******************************************************************************

************************ 1-3. Value Chain Scale Model *************************

*******************************************************************************

Parameter Level1(m,n);
*$CALL GDXXRW.EXE VCmake.xlsx par=Level1 rng=Sheet1!B2:F7
$GDXIN VCmake.gdx
$LOAD Level1
$GDXIN

Parameter Level2(n,m);
*$CALL GDXXRW.EXE VCuse.xlsx par=Level2 rng=Sheet1!B2:G6
$GDXIN VCuse.gdx
$LOAD Level2
$GDXIN

Parameter Level3(q,m);
*$CALL GDXXRW.EXE VCint.xlsx par=Level3 rng=Sheet1!B2:G4
$GDXIN VCint.gdx
$LOAD Level3
$GDXIN
* Once you import the excel file, you can disable $CALL commends.

Parameters
VCmake(m,n)      VC scale make matrix (VC process by VC product)
VCuse(n,m)       VC scale use matrix  (VC product by VC process)
VCmakeT(n,m)     VC scale make matrix transposed (VC product by VC process)
VCtech(n,m)      VC scale technology matrix (VC produce by VC process)
VCint(q,m)       VC scale intervention matrix (intervention by VC process)
;

VCmake(m,n) = Level1(m,n);
VCuse(n,m)  = Level2(n,m);

VCmakeT(n,m) = VCmake(m,n);

VCtech(n,m) = VCmakeT(n,m) - VCuse(n,m);



VCint(q,m) = Level3(q,m);
VCint('1','3') = 5;
VCint('1','4') =1;


*******************************************************************************

************************** 1-7. P2P Multiscale Model **************************

*******************************************************************************

Variable P2Px(j_k,i_k) P2P multiscale product and commodity transaction matrix;

Equations
P2Px17_18,P2Px17_22,

P2Px20_22
;




P2Px17_18(n,m).. P2Px(n,m) =e= VCtech(n,m);
P2Px17_22(n,o).. P2Px(n,o) =e= -EqUpCut(n,o);

P2Px.FX(p,m) = 0;
P2Px20_22(p,o).. P2Px(p,o) =e= POut;

Variable Fz Economic objective
         size_of_conventional_flow
         size_of_emergent_flow;


******************************************************************************

***************************Plug In Economy Scale******************************

******************************************************************************

Equation P2Px1_1,P2Px1_2,P2Px2_2,P2Px2_1;

P2Px.FX('1','1') = 0.5715;
P2Px.FX('1','2') = -0.143;
P2Px.FX('2','1') = -0.3469;
P2Px.FX('2','2') = 0.8393;

P2Px.fx(n,eu) = 0;
P2Px.fx(p,eu) = 0;

P2Px1_1.. P2Px('1','3') =e= -1.5*p0R1*0.2;
P2Px1_2.. P2Px('1','4') =e= -1.5*p0R1*0.2;

P2Px.FX('1','5') = 0;
P2Px.FX('1','6') = 0;
P2Px.FX('1','7') = 0;

P2Px2_1.. P2Px('2','3') =e= -2.0*p0R2*0.2;
P2Px2_2.. P2Px('2','4') =e= -2.0*p0R2*0.2;

P2Px.FX('2','5') = 0;
P2Px.FX('2','6') = 0;
P2Px.FX('2','7') = 0;

P2Px.fx('1','8') = 0;
P2Px.fx('2','8') = 0;

Variable P2Pint(q,i_k) P2P multiscale intervention matrix;

Equations
P2Pint18,P2Pint23
;

*The first two are for the economy scale
****************************Economy****************************
*$/kg
P2Pint.FX(q,'1') = 0.9;
P2Pint.FX(q,'2') = 2;

****************************VC*********************************
P2Pint18(q,m).. P2Pint(q,m) =e= VCint(q,m);

****************************Eq*********************************
P2Pint23(q,o).. P2Pint(q,o) =e= Eqint(q,o);


Variable P2Pg(q) P2P multiscale total interventions;

Equations P2P1,P2P2;

P2P1(j_k).. sum[i_k, P2Px(j_k,i_k) * P2Ps(i_k)] =e= P2Pf(j_k);
P2P2(q).. P2Pg(q) =e= sum[i_k, P2Pint(q,i_k) * P2Ps(i_k)];









Variables
CO2(i_k)   Total CO2 emissions from each sector and process
NOx(i_k)   Total NOx emissions from each sector and process
;

Equations CO2_1, NOx_1,CO2_2,CO2_3,CO2_4;

CO2_1(i_k).. CO2(i_k) =e= P2Pint('1',i_k) * P2Ps(i_k);
NOx_1(i_k).. NOx(i_k) =e= P2Pint('2',i_k) * P2Ps(i_k);

Variable Value_chain_emission,Equipment_scale_emission,Economy_em;

CO2_2.. Value_chain_emission =E= CO2('3')+CO2('4')+CO2('5')+CO2('6') + CO2('7');
CO2_3.. Equipment_scale_emission =E= CO2('8');

CO2_4.. Economy_em =E= CO2('1') + CO2('2');





**Calculating the exact final demand to the economy for comparison***********************************************************************************************************************************************************************************************************************************************************************************************************

Equation final_demand1,final_demand2;
variable demand_to_PR1,demand_to_PR2;
final_demand1.. demand_to_PR1/0.2 =E= p2ps('3')*1.5 + p2ps('4')*1.5;
final_demand2.. demand_to_PR2/0.2 =E= p2ps('3')*2 + p2ps('4')*2;




*************************************************************

*** Economic objective to minimize total economy factor costs

*************************************************************



Equation size_determination1,size_determination2;
*Equation size_determination2 ;
size_determination1.. size_of_conventional_flow =e= F1 * p2ps('8')-size_of_emergent_flow;
size_determination2.. size_of_emergent_flow *(p2ps('3')+p2ps('4')) =e= F1 * p2ps('8') * p2ps('4');


********************Economic calculations**************************************************************************************************************************************


Variable cost;
*Yearly Cost*
Eng5.. cost =e= (p_F1_1 * size_of_conventional_flow + p_F1_2 * size_of_emergent_flow + p_F2 * F2 * p2ps('5'));
Eng6.. Profit =e= p2ps('8') * pOUt * 40 - cost;



Variable c1,c2,c3;
Equation cc1,cc2,cc3;
cc1.. c1 =e=  size_of_conventional_flow * p_F1_1;
cc2.. c2 =e=  size_of_emergent_flow * p_F1_2;
cc3.. c3 =e=  p2ps('5') * F2* p_F2;



*****************************************************************************************************************************************************************************************Evolution of Contraints of Production from CGE functions**********************************************************************************************************************


Equation constraints1;

constraints1.. (p2ps('3')**0.715) * (p2ps('4')**0.69) =e= 1;


*******************************************************************************

*************************** 3. Objective Functions ****************************

*******************************************************************************

*** Environmental objective to minimize total CO2 emissions
Variable Ez1,Ez2,Ez3,Ez4 Environmental objective from the Equipment scale and the value chain scale;


Equation Eobj1,Eobj2,Eobj3,Eobj4;

Eobj1.. Ez1 =e= Equipment_scale_emission;
Eobj2.. Ez2 =e= Value_chain_emission+Equipment_scale_emission;
Eobj3.. Ez3 =e= P2Pg('1');
Eobj4.. Ez4 =e= env_g;





*************************** First Equilibrium Condition*************************

*************************** Only Conventional Technology Exists****************

****Constraint to model such a state




pOUT.FX = 1;

*******************************************************************************

************************* 4. Optimization Formulation *************************

*******************************************************************************

Model CGEP2P /all/;
CGEP2P.optfile = 1;


Option NLP = %mydata%;
option bRatio = 1;


Set ANSWER /OUTPUT_OF_GAMS_CODE/;


*******************************Writing Header in Output File*******************************************************


file fx /%myfile%/;
fx.ps = 200;
fx.pw = 30000;
fx.nd = 2;
fx.pc=8;
fx.ap = 1;
put fx;
put 'Fn_dm'                     ;
put 'Tax'                    ;
put 'Eqem'                       ;
put 'EqVC'                       ;
put 'Tot'                       ;
put 'Rae'                        ;
put 'Eff'                        ;
put 'Cos'                         ;
put 'Conv_Flow'                   ;
put 'Emer_Flow'                   ;

put 'F2'                          ;
put 'F1'
put 'C1'
put 'C2'
put 'C3'
put 'Price_Conv_flow'                     ;
put 'Price_Emergent_flow'                     ;
put 'Price F2' ;
putclose;








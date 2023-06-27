from scipy import integrate
def double_bracket(f):
    # Compute the double bracket of a function
    mean = integrate.quad(f,0,1)[0]
    brace = lambda y: f(y)-mean
    brack_nzm = lambda y: integrate.quad(brace,0,y)[0]
    mean_bracket = integrate.quad(brack_nzm,0,1)[0]
    def brack(y):
        return integrate.quad(brace,0,y)[0] - mean_bracket
    return brack

def C_values(a, dady, delta):
    ainvsquared = lambda y: 1/a(y)**2
    ainv = lambda y: 1/a(y)
    ay_ainv = lambda y: ainvsquared(y)*dady(y)
    #-<a[[ay a^-2]]> = <ay a^-2 [[a]]>
    db = double_bracket(a)
    integrand = lambda y: ay_ainv(y)*db(y)
    C1=(1/delta)*integrate.quad(integrand,0,1)[0]

    #<a[[ay a^-2]]>
    db = double_bracket(ay_ainv)
    integrand = lambda y: a(y)*db(y)
    C11=(1/delta)*integrate.quad(integrand,0,1)[0]

    #<a[[[[ay a^-2]]]]>
    db = double_bracket(ay_ainv)
    dbdb = double_bracket(db)
    integrand = lambda y: a(y)*dbdb(y)
    C2 =(1/delta)*integrate.quad(integrand,0,1)[0]

    #<a[[ay a^-2 [[a]]]]>
    db = double_bracket(a)
    aydb= lambda y: ay_ainv(y)*db(y)
    dbdb = double_bracket(aydb)
    integrand = lambda y: a(y)*dbdb(y)
    C3 =(1/delta)*integrate.quad(integrand,0,1)[0]

    #<a[[[[a]]]]> not equal to zero 
    db = double_bracket(a)
    dbdb = double_bracket(db)
    integrand = lambda y: a(y)*dbdb(y)
    C4 =(1/delta)*integrate.quad(integrand,0,1)[0]

    #<[[a]]**2 [[ay a^-2 ]] >
    db = double_bracket(ay_ainv)
    db_a = double_bracket(a)
    inside_brackets = lambda y: db(y) * db_a(y)**2
    C5 = (1/delta)*integrate.quad(inside_brackets,0,1)[0]

    #<ay a^-2[[a[[ay a^-2 ]]]]>
    db = double_bracket(ay_ainv)
    aydb= lambda y: a(y)*db(y)
    dbdb = double_bracket(aydb)
    integrand = lambda y: ay_ainv(y)*dbdb(y)
    C6 =(1/delta)*integrate.quad(integrand,0,1)[0]

    #<ay a^-2 [[[[a]]]]>
    db = double_bracket(a)
    dbdb = double_bracket(db)
    integrand = lambda y: ay_ainv(y)*dbdb(y)
    C7 =(1/delta)*integrate.quad(integrand,0,1)[0]

    #<[[a]]^2 ay a^-2>
    db = double_bracket(a)
    integrand = lambda y: ay_ainv(y) * db(y)**2 
    C8 =(1/delta)*integrate.quad(integrand,0,1)[0]

    #<a>^2
    a2 = lambda y: a(y)**2
    avga=(1/delta)*integrate.quad(a,0,1)[0]
    avga2= (1/delta)*integrate.quad(a2,0,1)[0]
    ainvavg = (1/delta)*integrate.quad(ainv,0,1)[0]
    return C1, C2, C3, C4, C5, C6, C7, C8, avga,ainvavg
    #print("<a>^2= ",avga**2,"<a^2>= ", avga2)
    #print("c1= " , C1)
    #print("c2= " , C2)
    #print("c3= " , C3)
    #print("c4= " , C4)
    #print("c5= " , C5)
    #print("c6= " , C6)
    #print("c7= " , C7, C7-C2)
    #print("c8= " , C8)
    #print("c11= " , C11, C1+C11)
    
def Homogenized_system_coef(C1, C2, C3, C4, C5, C6, C7, C8, avga,ainvavg, p_0, P1, P11, delta):
    r1 = -2*C1/p_0 
    r2 = ((-3*C1+4*C1**2)/(avga*P1*p_0**2) + (C1*P11/(avga*p_0*P1**2)))
    r3 = 2*delta*((-avga*C2+C3)/(avga*p_0))
    r4 = 2*delta*(-C2/p_0 + C3/(avga*p_0)) 
    r5 = 2*C1/(p_0**2)
    r6 = delta**2 *C4/(avga**2)
    k1 = -P1 
    k2 = 2*(C1-1)/(avga*p_0) 
    k3 = 2*(1-C1)/(avga*p_0**2) 
    k4 = 4*(-C1+C1**2)/(avga*p_0**2) + (4*C6+ainvavg)/(p_0**2)
    k5 = -P11 
    k6 = -C3*delta/(avga**2 *p_0 )
    k7 = -2*delta*C2/(avga*p_0)
    return r1, r2, r3, r4, r5, r6, k1, k2, k3, k4, k5, k6, k7
        
# FUNKCIJAS MINIMUMA ATRAŠANA AR GRADIENTA METODI

# SymPy dokumentācija:
# https://docs.sympy.org/latest/index.html

# SciPy dokumentācija:
# https://docs.scipy.org/doc/scipy/

# Ideja par skaitļu noapaļošanu atbilstoši precizitātei:
# https://stackoverflow.com/questions/41020797/proper-rounding-in-python-to-3-decimal-places

# Ideja par datu 3D vizualizāciju ar matplotlib:
# https://stackoverflow.com/questions/35363444/plotting-lines-connecting-points

from decimal import ROUND_HALF_UP, Decimal
import numpy as np
from sympy import symbols, diff, log, exp
import matplotlib.pyplot as plt

def solve_gradient(max_iter, t_star, epsi, coef, x_prim_beg):
    # reģistrēt atrisinājuma mainīgos
    x1, x2, x3 = symbols("x1 x2 x3")
    
    # definēt funkcijas koeficientus atbilstoši variantam 11
    a, b, c, d, e, f = coef
    
    # definēt pētāmo funkciju
    func = a*x1**4 - b*x1*x2**2 + c*x2**2*x3**2 - \
        d*x3**3 + e*x1 - f*x2 + exp(x3) - \
        log(x1**2+x2**2+1)
    
    # piešķirt punkta x sākumvērtību
    x_prim = x_prim_beg
    
    # definēt iterāciju un funkcijas vērtību masīvus
    iterations = np.array([])
    function_values = np.array([])
    
    # definēt atrisinājuma koordināšu masīvus
    x1_points = np.array([])
    x2_points = np.array([])
    x3_points = np.array([])
    
    # definēt gradienta normu masīvu
    norms = np.array([])
    
    # atrast funkcijas parciālos atvasinājumus
    x1_partial = diff(func, x1)
    x2_partial = diff(func, x2)
    x3_partial = diff(func, x3)
    
    for iter in range(1, max_iter):
        # noteikt funkcijas mainīgiem atbilstošus argumentus
        x_values = {x1: x_prim[0], x2: x_prim[1], x3: x_prim[2]}
        # noteikt gradienta vērtību punktā x_prim
        grad = [float(x1_partial.subs(x_values)),
                float(x2_partial.subs(x_values)),
                float(x3_partial.subs(x_values))]
        
        # aizvietot funkcijas mainīgos ar argumentiem
        value = func.subs(x_values)
    
        # papildināt sarakstus ar iterācijām un funkcijas vērtībām 
        iterations = np.append(iterations, iter)
        function_values = np.append(function_values, value)
        
        # papildināt sarakstus ar iterācijā pieejamiem risinājumiem 
        x1_points = np.append(x1_points, x_prim[0])
        x2_points = np.append(x2_points, x_prim[1])
        x3_points = np.append(x3_points, x_prim[2])
        
        # noteikt gradienta normu
        norm = np.linalg.norm(grad)
        
        # papildināt sarakstu ar normām
        norms = np.append(norms, norm)
        
        # pārbaudīt, vai nav sasniegta precizitāte
        if norm <= epsi:
            break
        
        # atrast jaunu tuvinātu atrisinājumu
        x_prim = list(map(lambda x_prim, grad:
            x_prim - t_star*grad, x_prim, grad))
    
    # konstruēt 3D grafiku, kas attēlo atrisinājuma
    # konverģenci uz optimālo rezultātu
    figure = plt.figure(figsize=(8, 6))
    axes1 = figure.add_subplot(111, projection='3d')
    axes1.scatter(x1_points, x2_points, x3_points,
                c='b', marker='o')
    axes1.set_title('Att.1: Atrisinājuma konverģence uz optimumu')
    axes1.set_xlabel('x1')
    axes1.set_ylabel('x2')
    axes1.set_zlabel('x3')

    # Realizēt punktu savienošanu ar līniju
    for i in range(0, x1_points.size-1):
        axes1.plot([x1_points[i], x1_points[i+1]],
                   [x2_points[i], x2_points[i+1]],
                   [x3_points[i], x3_points[i+1]],
                   color='b', linestyle='-', linewidth=2)

    plt.show()
    
    # konstruēt grafiku, kas attēlo izrēķināto
    # funkcijas vērtību atkarībā no iteraciju skaita
    plt.plot(iterations, function_values)
    axes2 = plt.subplot()
    axes2.set_title('Att.2: Funkcijas vērtība atkarībā no iterāciju skaita')
    axes2.set_xlabel('Iterācijas')
    axes2.set_ylabel('Funkcijas vērtība')
    plt.show()
    
    # konstruēt grafiku, kas attēlo normas
    # vērtību atkarībā no iteraciju skaita
    plt.plot(iterations, norms)
    axes3 = plt.subplot()
    axes3.set_title('Att.3: Normas vērtība atkarībā no iteraciju skaita')
    axes3.set_xlabel('Iterācijas')
    axes3.set_ylabel('Normas vērtība')
    plt.show()
    
    # izvadīt atrastās funkcijas minimizētās vērtības
    # ar precizitāti, ko nosaka mainīgais epsi
    print('-----------------------------------------------')
    print('Optimization results (with given epsi precision):')
    print('x1: ', Decimal(str(x_prim[0])).quantize(Decimal(str(epsi)), rounding=ROUND_HALF_UP))
    print('x2: ', Decimal(str(x_prim[1])).quantize(Decimal(str(epsi)), rounding=ROUND_HALF_UP))
    print('x3: ', Decimal(str(x_prim[2])).quantize(Decimal(str(epsi)), rounding=ROUND_HALF_UP))
    print('-----------------------------------------------')
    
    # izvadīt veikto iterāciju skaitu un funkcijas vērtību
    print('Iterations: ', iter)
    print('Function value: ', Decimal(str(value)).quantize(Decimal(str(epsi)), rounding=ROUND_HALF_UP))
    print('-----------------------------------------------')
    
# definēt sākuma parametrus
max_iter = 10_000
t_star = 0.01
epsi = 0.001

coef = [1, 5, -4, 5, 5, -4]
x_prim_beg = [0.0, 0.0, 0.0]

# izsaukt optimuma meklēšanas funkciju
solve_gradient(max_iter, t_star, epsi, coef, x_prim_beg)

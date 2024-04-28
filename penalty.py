# FUNKCIJAS MINIMUMA ATRAŠANA AR SODA METODI

# IOP 4.praktiskās nodarbības slaidi:
# https://estudijas.rtu.lv/mod/resource/view.php?id=4359586

# SciPy dokumentācija:
# https://docs.scipy.org/doc/scipy/

# Ideja par skaitļu noapaļošanu atbilstoši precizitātei:
# https://stackoverflow.com/questions/41020797/proper-rounding-in-python-to-3-decimal-places

# Ideja par datu 3D vizualizāciju ar matplotlib:
# https://stackoverflow.com/questions/35363444/plotting-lines-connecting-points

from decimal import ROUND_HALF_UP, Decimal
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

def call_function(args, coef):
    # definēt funkcijas argumentus
    x1, x2, x3 = args
    
    # definēt funkcijas koeficientus atbilstoši variantam 11
    a, b, c, d, e, f = coef
    
    # definēt pētāmo funkciju
    func = a*x1**4 - b*x1*x2**2 + c*x2**2*x3**2 - \
        d*x3**3 + e*x1 - f*x2 + np.exp(x3) - \
        np.log(x1**2+x2**2+1)
    
    return func

def calculate_penalty(args, coef, r):
    # definēt funkcijas argumentus
    x1, x2, x3 = args
    
    # noteikt sodu
    penalty = r * (
        max(0, -x1 - x2 - 5)**2 +
        max(0, -x2 - x3 - 1)**2 +
        max(0, -x1 - 6)**2 +
        max(0, x1)**2 +
        max(0, x2)**2 +
        max(0, x3)**2  
    )
    
    return call_function(args, coef) + penalty

def solve_penalty(max_iter, r, epsi, coef, x0):
    # definēt iterāciju un funkcijas vērtību masīvus
    iterations = np.array([])
    function_values = np.array([])
    
    # definēt atrisinājuma koordināšu masīvus
    x1_points = np.array([])
    x2_points = np.array([])
    x3_points = np.array([])
    
    # piešķirt iepriekšējam atrisinājumam tukšumu
    prev_solution = None
    
    for iter in range(1, max_iter):
        # minimizēt soda funkciju
        cur_solution = minimize(
            lambda args: calculate_penalty(args, coef, r),
            x0
        )
        
        # noteikt tuvināta atrisinājuma funkcijas vērtību
        value = cur_solution.fun
        
        # papildināt sarakstus ar iterācijām un funkcijas vērtībām 
        iterations = np.append(iterations, iter)
        function_values = np.append(function_values, value)
        
        # papildināt sarakstus ar iterācijā pieejamiem risinājumiem 
        x1_points = np.append(x1_points, cur_solution.x[0])
        x2_points = np.append(x2_points, cur_solution.x[1])
        x3_points = np.append(x3_points, cur_solution.x[2])
        
        # pārbaudīt, vai nav sasniegta precizitāte
        if prev_solution is not None and \
            np.abs(cur_solution.fun - prev_solution.fun) <= epsi:
            break
        
        # atzīt esošo atrisinājumu par iepriekšējo
        prev_solution = cur_solution
        # palielināt soda koeficientu
        r *= 10
    
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
    
    # izvadīt atrastās funkcijas minimizētās vērtības
    # ar precizitāti, ko nosaka mainīgais epsi
    print('-----------------------------------------------')
    print('Optimization results (with given epsi precision):')
    print('x1: ', Decimal(str(cur_solution.x[0])).quantize(Decimal(str(epsi)), rounding=ROUND_HALF_UP))
    print('x2: ', Decimal(str(cur_solution.x[1])).quantize(Decimal(str(epsi)), rounding=ROUND_HALF_UP))
    print('x3: ', Decimal(str(cur_solution.x[2])).quantize(Decimal(str(epsi)), rounding=ROUND_HALF_UP))
    print('-----------------------------------------------')
    
    # izvadīt veikto iterāciju skaitu un funkcijas vērtību
    print('Iterations: ', iter)
    print('Function value: ', Decimal(str(value)).quantize(Decimal(str(epsi)), rounding=ROUND_HALF_UP))
    print('-----------------------------------------------')
    
# definēt sākuma parametrus
max_iter = 10_000
epsi = 0.001
r = 1

coef = [1, 5, -4, 5, 5, -4]
x0 = [0.0, 0.0, 0.0]

# izsaukt optimuma meklēšanas funkciju
solve_penalty(max_iter, r, epsi, coef, x0)

import numpy as np
from time import process_time

def integrate(x, y, method, N):
    integral = []
    for t in range(1, N):
        integral.append(method(y=y[0:t], x=x[0:t]))
    return np.array(integral)


def torch_multiple_integrate(x, f_n, method, N):
    integral = []
    x_range = np.arange(x[0], x[1], (x[1] - x[0])/N)
    for t in range(2, N):
        x_t = [[x_range[0], x_range[t]]]
        integral.append(method.integrate(fn=f_n, N=t, integration_domain=x_t))
    return np.array(integral)  

def runtime_measure(x, y, method, N_steps=10, N_average=1):
    runtime_measurement=[]
    for n in range(N_steps):
        measure = 0
        for m in range(N_average):
            start_time = process_time()
            integrate(x, y, method, n)
            stop_time = process_time()
            measure = measure + stop_time - start_time
        runtime_measurement.append(measure/N_average)
    
    return np.array(runtime_measurement)
        

def torch_runtime_measure(x, f_n, method, N_points=10, N_average=1):
    runtime_measurement=[]
    for n in range(2,N_points):
        measure = 0
        for m in range(N_average):
            start_time = process_time()
            torch_multiple_integrate(x, f_n, method, n)
            stop_time = process_time()
            measure = measure + stop_time - start_time
        runtime_measurement.append(measure/N_average)
    
    return np.array(runtime_measurement)
        

# Function that provides the iterative integrals

def iter_int(u,t0, tf, dt, Ntrunc):
    import numpy as np
    
    """
    Returns the list of all iterated integrals of the input u indexed by the words of length from 1 to Ntrunc.
    
    
    Parameters:
    -----------
    u: array_like
        The array of input functions u_i: [t0, tf] -> IR, for all i in {1, ..., m}
        stacked vertically. Each function has the form u_i = np.array([u_i[0], u_i[1], ..., u_i[N]])  
        where u_i[0] = u_i(t_0), u_i[1] = u_i(t_0+dt), ..., u_i[N] = u_i(tf), N = int((tf-t0)//dt+1)
        and u = np.vstack([u_1, u_2, ..., u_m])
    
    t0: float
        Initial point of the time-interval domain of the inputs
    
    tf: float
        Final time of the time-interval domain of the inputs
    
    dt: float
        The size of the step of the evenly spaced partition of the time-interval domain 
    
    Ntrunc: int
        The truncation length of the words of the Chen-Fliess series
        Sum_{i=0}^Ntrunc Sum_{eta in X^i} (c, eta) E_{eta}[u](t0,tf)
    
    
    
    Returns:
    --------
    list: ndarray
    
    [
    [E_{x_0}[u](t0, t0), E_{x_0}[u](t0, t0+dt), ..., E_{x_0}[u](t0, tf)],
    [E_{x_1}[u](t0, t0), E_{x_1}[u](t0, t0+dt), ..., E_{x_1}[u](t0, tf)],
     .
     .
     .
    [E_{x_m}[u](t0, t0), E_{x_m}[u](t0, t0+dt), ..., E_{x_m}[u](t0, tf)],
    ---------------------------------------------------------------------------------------
    [E_{x_0x_0}[u](t0, t0), E_{x_0x_0}[u](t0, t0+dt), ..., E_{x_0x_0}[u](t0, tf)],
    [E_{x_1x_0}[u](t0, t0), E_{x_1x_0}[u](t0, t0+dt), ..., E_{x_1x_0}[u](t0, tf)],
     .
     .
     .
    [E_{x_mx_0}[u](t0, t0), E_{x_mx_0}[u](t0, t0+dt), ..., E_{x_mx_0}[u](t0, tf)],
    [E_{x_0x_1}[u](t0, t0), E_{x_0x_1}[u](t0, t0+dt), ..., E_{x_0x_1}[u](t0, tf)],
    [E_{x_1x_1}[u](t0, t0), E_{x_1x_1}[u](t0, t0+dt), ..., E_{x_1x_1}[u](t0, tf)],
     .
     .
     .
    [E_{x_mx_1}[u](t0, t0), E_{x_mx_1}[u](t0, t0+dt), ..., E_{x_mx_1}[u](t0, tf)],
     .
     .
     .
    [E_{x_0x_m}[u](t0, t0), E_{x_0x_m}[u](t0, t0+dt), ..., E_{x_0x_m}[u](t0, tf)],
    [E_{x_1x_m}[u](t0, t0), E_{x_1x_m}[u](t0, t0+dt), ..., E_{x_1x_m}[u](t0, tf)],
     .
     .
     .
    [E_{x_mx_m}[u](t0, t0), E_{x_mx_m}[u](t0, t0+dt), ..., E_{x_mx_m}[u](t0, tf)],
    ----------------------------------------------------------------------------------------
    [E_{x_0x_0x_0}[u](t0, t0), E_{x_0x_0x_0}[u](t0, t0+dt), ..., E_{x_0x_0x_0}[u](t0, tf)],
    [E_{x_0x_1x_0}[u](t0, t0), E_{x_0x_1x_0}[u](t0, t0+dt), ..., E_{x_0x_1x_0}[u](t0, tf)],
     .
     .
     .
    [E_{x_0x_mx_0}[u](t0, t0), E_{x_0x_mx_0}[u](t0, t0+dt), ..., E_{x_0x_mx_0}[u](t0, tf)],
     .
     .
     .
    [E_{x_mx_mx_m}[u](t0, t0), E_{x_mx_mx_m}[u](t0, t0+dt), ..., E_{x_mx_mx_m}[u](t0, tf)]
    ----------------------------------------------------------------------------------------
     .
     .
     .
    ----------------------------------------------------------------------------------------
    [E_{x_0...x_0}[u](t0, t0), E_{x_0...x_0}[u](t0, t0+dt), ..., E_{x_0...x_0}[u](t0, tf)],
     .
     .
     .
    [E_{x_m...x_m}[u](t0, t0), E_{x_m...x_m}[u](t0, t0+dt), ..., E_{x_m...x_m}[u](t0, tf)],
    ]
    
    
    
    Examples:
    ---------
    
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    # Define the system
    def system(t, x, u1_func, u2_func):
        x1, x2 = x
        u1 = u1_func(t)
        u2 = u2_func(t)
        dx1 = -x1*x2 +  x1 * u1
        dx2 = x1*x2 - x2* u2
        return [dx1, dx2]

    # Input 1
    def u1_func(t):
        return np.sin(t)

    # Input 2
    def u2_func(t):
        return np.cos(t)

    # Initial condition
    x0 = [1/6,1/6]

    # Time range
    t_span = (0, 3)

    # Solve the ODE with dense_output=True
    solution = solve_ivp(system, t_span, x0, args=(u1_func, u2_func), dense_output=True)

    # Partition of the time range
    t = np.linspace(t_span[0], t_span[1], 300)
    y = solution.sol(t)

    # Plot the results
    plt.plot(y[0].T, y[1].T)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    plt.show()

    plt.plot(t, y[0].T)
    plt.xlabel('Time')
    plt.ylabel('$x_1$')
    plt.grid()
    plt.show()
    
    """
    
    
    if t0 < 0:
        raise ValueError("The initial point, %s, must be non-negative." %t0)
        
    if tf < 0:
        raise ValueError("The final point, %s, must be non-negative." %tf)
        
    if dt < 0:
        raise ValueError("The step, %s, must be non-negative." %dt)
        
    if Ntrunc < 1:
        raise ValueError("The truncation length, %s, must be non-negative." %Ntrunc)
        
    if not isinstance(Ntrunc, int):
        raise ValueError("The truncation length, %s, must be an integer." %Ntrunc)
        
    if not isinstance(u, np.ndarray):
        raise ValueError("The input, %s, must be an ndarray." %u)
        
    if u.shape[0] == 0:
        raise ValueError("The number of rows, %s, must be greater than zero." %u.shape[0])
        

        
        
    # The length of the partition of time is computed
    length_t = int((tf-t0)//dt+1)
    
    if u.shape[1] != length_t:
        raise ValueError("The length of the input, %s, must be int((tf-t0)//dt+1) = %s." %(u.shape[1], length_t))
    
    # The sequence of numbers that represent the partition of the time domain of the input is generated.
    # This is used to discretize the integrals.
    t = np.linspace(t0, tf, length_t)
    # The input u_0 associated with the letter x_0 is generated.
    u0 = np.ones(length_t)
    # The inputs of the system associated with x_1, ..., x_m with the input associated to x_0 vertically are stacked.
    # [
    #  [u_0(t0), u_0(t0+dt), ..., u_0(tf)],
    #  [u_1(t0), u_1(t0+dt), ..., u_1(tf)],
    #    .
    #    .
    #    .
    #  [u_m(t0), u_m(t0+dt), ..., u_m(tf)]
    # ]
    u = np.vstack([u0, u])
    
    # The number of rows which are equal to the number of total input functions is obtained.
    num_input = int(np.size(u,0))

    # The total number of iterated integrals of word length less than or equal to the truncation length is computed.
    # total_iterint = num_input + num_input**2 + ... + num_input**Ntrunc
    if num_input == 1:
    total_iterint = Ntrunc
else:
    total_iterint = num_input * (1 - pow(num_input, Ntrunc)) // (1 - num_input)
    # Handle special case where num_input = 1 (to avoid division by zero)
     #if num_input == 1:
     #total_iterint = Ntrunc
     #else:
     #total_iterint = num_input * (1 - pow(num_input, Ntrunc)) // (1 - num_input)

    # This is transformed into an integer.
    total_iterint = int(total_iterint)
    
    # A matrix of zeros with as many rows as the total number of iterated integrals and as many columns
    # as the elements in the partition of time is computed.
    Etemp = np.zeros((total_iterint,length_t))
    # Starts the list ctrEtemp such that ctrEtemp[k]-ctrEtemp[k-1] = the number of iterated integrals 
    # of word length k
    # ctrEtemp[0] = 0
    # ctrEtemp[k] = num_input + num_input**2 + ... + num_input**k,  1<=k<=Ntrunc
    ctrEtemp = np.zeros(Ntrunc+1)

    # ctrEtemp[k] = num_input + num_input**2 + ... + num_input**k,  1<=k<=Ntrunc
    for i in range(Ntrunc):
        ctrEtemp[i+1] = ctrEtemp[i]+pow(num_input,i+1)
    
    
    # The iterative integrals of the words of length 1, E_{x_i}[u](t0, tf) for all i in {0, ..., m}, are computed.
    
    # First, E_{x_i}[u](t0, tf) for all i in {0, ..., m} are computed 
    # for all tf neq t0
    sum_acc = np.cumsum(u, axis = 1)*dt
    # Then the values of E_{x_i}[u](t0, tf) for tf = t0, this is E_{x_i}[u](t0, t0) = 0, are added
    # to have E_{x_i}[u](t0, tf) for all tf>=t0
    Etemp[:num_input,:] = np.hstack((np.zeros((num_input,1)), sum_acc[:,:-1]))
    
    
    # The iterated integrals of the words of length k => 1, E_{x_{i_1}...x_{i_k}}[u](t0, tf) for all i_j in {0, ..., m}, 
    # are computed at each iteration.
    
    for i in range(1,Ntrunc):
        # start_prev_block = num_input + num_input**2 + ... + num_input**(i-1)
        start_prev_block = int(ctrEtemp[i-1])
        # end_prev_block = num_input + num_input**2 + ... + num_input**i
        end_prev_block = int(ctrEtemp[i])
        # end_current_block = num_input + num_input**2 + ... + num_input**(i+1)
        end_current_block = int(ctrEtemp[i+1])
        # num_prev_block = num_input**i
        num_prev_block = end_prev_block - start_prev_block
        # num_current_block = num_input**(i+1)
        num_current_block = end_current_block - end_prev_block
        
        """ 
        U_block =
        [
         u_0
         u_0
          .   
          .   # u_0 repeats num_input**i times
          . 
         u_0
         u_1
         u_1
          .
          .   # u_1 repeats num_input**i times
          .
         u_1
          .
          .   # u_k repeats num_input**i times
          .
         u_m
         u_m
          .
          .   # u_m repeats num_input**i times
          .
         u_m
        ]
        """
        U_block = u[np.repeat(range(num_input), num_prev_block), :]
        
        """
        prev_int_block =
        [
        E_{x_0...x_0x_0}[u](t0,tf)  
        E_{x_0...x_1x_0}[u](t0,tf)
                .
                .                         # block with all the num_input**i iterated integrals of words of length i
                .
        E_{x_m...x_mx_m}[u](t0,tf)
        --------------------------
        E_{x_0...x_0x_0}[u](t0,tf)  
        E_{x_0...x_1x_0}[u](t0,tf)
                .
                .                         # block with all the num_input**i iterated integrals of words of length i
                .
        E_{x_m...x_mx_m}[u](t0,tf)
        --------------------------
                .
                .
                .
        --------------------------
        E_{x_0...x_0x_0}[u](t0,tf)  
        E_{x_0...x_1x_0}[u](t0,tf)
                .
                .                         # block with all the num_input**i iterated integrals of words of length i
                .
        E_{x_m...x_mx_m}[u](t0,tf)
        ]
        
        In total there are num_input blocks 
        """
        prev_int_block = np.tile(Etemp[start_prev_block:end_prev_block,:],(num_input,1))
        
        
        """
        U_block*prev_int_block = 
        [
        u_0(tf)E_{x_0...x_0x_0}[u](t0,tf)  
        u_0(tf)E_{x_0...x_1x_0}[u](t0,tf)
                .
                .                         # block with all the num_input**i iterated integrals of words of length i
                .
        u_0(tf)E_{x_m...x_mx_m}[u](t0,tf)
        --------------------------
        u_1(tf)E_{x_0...x_0x_0}[u](t0,tf)  
        u_1(tf)E_{x_0...x_1x_0}[u](t0,tf)
                .
                .                         # block with all the num_input**i iterated integrals of words of length i
                .
        u_1(tf)E_{x_m...x_mx_m}[u](t0,tf)
        --------------------------
                .
                .
                .
        --------------------------
        u_m(tf)E_{x_0...x_0x_0}[u](t0,tf)  
        u_m(tf)E_{x_0...x_1x_0}[u](t0,tf)
                .
                .                         # block with all the num_input**i iterated integrals of words of length i
                .
        u_m(tf)E_{x_m...x_mx_m}[u](t0,tf)
        ]
        
        current_int_block integrates U_block*prev_int_block
        """
        current_int_block = np.cumsum(U_block*prev_int_block, axis = 1)*dt
        # Stacks the block of iterated integrals of word length i+1 into Etemp
        Etemp[end_prev_block:end_current_block,:] = np.hstack((np.zeros((num_current_block,1)), current_int_block[:,:-1]))

    itint = Etemp
    return itint                                                                        



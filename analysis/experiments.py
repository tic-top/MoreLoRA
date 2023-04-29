import matplotlib.pyplot as plt
import numpy as np

def draw_h(type = 'mul'):
    # Define values for r
    r_values = range(1, 9)

    # Define values for k
    k_values = range(1, 5)

    # Loop through k values and plot (r/k)^k for each value
    for k in k_values:
        if type=='mul':
            y_values = [(r / k) ** k for r in r_values]
        else:
            y_values = [(2 * r / k) ** k for r in r_values]
        plt.plot(r_values, y_values, label=f"k={k}")

    # Add labels and legend to the plot
    plt.xlabel("r")
    if (type=='mul'):
        plt.ylabel("rank=$(\\frac{r}{k})^k$")
    else:
        plt.ylabel("rank=$(\\frac{2r}{k})^k$")
    plt.legend()

    # Show the plot
    plt.show()

def experiment(type = 'mul', 
               m = 512, 
               n = 256, 
               r = 1, 
               k = 1, 
               regularization = False,
               num_experiments = 10, 
               verbose=False):
    """
    type:
    mul,    add,
    h-mul,  h-add
    """
    ranks = np.zeros(num_experiments)

    def lora(type):
        if type == "h_mul" or type =="mul":
            if not (r % k == 0):
                raise Exception("r must be divisible by k")
            rr = r // k
            if not (m % rr == 0 and n % rr==0):
                raise Exception("m and n must be divisible by r/k")
            print(f"est rank: {min(m,n,(rr)**k)}")
            def lora_h_mul():
                U = np.random.normal(size=(m, r))
                V = np.random.normal(size=(n, r))
                # Computer W
                for j in range(k):
                    U1= U[:, j * rr : (j+1) * rr]
                    V1= V[:, j * rr : (j+1) * rr]
                    # Wi= U1.dot(Iu) + Iv.dot(V1.T)
                    Wi = U1.dot(V1.T)
                    if j==0:
                        W = Wi
                    else:
                        W *= Wi
                # Compute the rank of the W matrix
                rank = np.linalg.matrix_rank(W)
                return rank
            return lora_h_mul
        elif type == 'h_add' or type == "add":
            if not (r % k == 0):
                raise Exception("r must be divisible by k")
            rr = r // k
            if not (m % rr == 0 and n % rr==0):
                raise Exception("m and n must be divisible by r/k")
            print(f"est rank: {min(m,n,(2*rr)**k)}")
            
            I = np.eye(rr)
            Iu = np.vstack([I] * (n // rr)).T
            Iv = np.vstack([I] * (m // rr))
            def lora_h_add():
                U = np.random.normal(size=(m, r))
                V = np.random.normal(size=(n, r))
                # Computer W
                for j in range(k):
                    U1= U[:, j * rr : (j+1) * rr]
                    V1= V[:, j * rr : (j+1) * rr]
                    Wi= U1.dot(Iu) + Iv.dot(V1.T)
                    # print(np.linalg.matrix_rank(Wi))
                    if j==0:
                        W = Wi
                    else:
                        W *= Wi
                # Compute the rank of the W matrix
                rank = np.linalg.matrix_rank(W)
                # print(rank)
                return rank
            return lora_h_add

    try:   
        func = lora(type)
        print(f"trainbale: {r * (m+n) / (m*n)}")
        for i in range(num_experiments):
            ranks[i] = func()
        print(f"exp rank: {ranks[0]}")
        if verbose:
            # Display the histogram of ranks
            plt.hist(ranks,bins=range(max(1,-50+int(ranks[0]) ), 1+min(m,n,50+int(ranks[0]))), density=True)
            plt.xlabel('Rank')
            plt.ylabel('Frequency')
            plt.show()
    except Exception as e:
        print(e.__str__())
    
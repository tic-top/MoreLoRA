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
               rr = 1, 
               regularization = False,
               num_experiments = 10, 
               verbose=False):
    """
    type:
    mul,    add,
    h-mul,  h-add
    """
    ranks = np.zeros(num_experiments)

    def lora(type, r, rr):
        if type == 'mul' or type == 'add':
            rr = r
        k = max(1, r//rr+1)
        if 'mul' in type:
            def lora_h_mul():
                U = np.random.normal(size=(m, r))
                V = np.random.normal(size=(r, n))
                cnt = 0
                # Computer W
                for j in range(k):
                    if j*rr < r:
                        Ui= U[:, j * rr : min(r,(j+1) * rr)]
                        Vi= V[j * rr : min(r,(j+1) * rr), :]
                        if j==0:
                            W = Ui.dot(Vi)
                        else:
                            W *= Ui.dot(Vi)
                        cnt += 1
                    else:
                        break
                # print(cnt)
                return np.linalg.matrix_rank(W)
            return lora_h_mul
        else:
            def lora_h_add():
                U = np.random.normal(size=(m, r))
                V = np.random.normal(size=(r, n))
                cnt = 0
                for j in range(k):
                    if j*rr < r:
                        lb = j * rr
                        ub = min(r,(j+1) * rr)
                        delta = ub - lb
                        Ui= U[:, lb : ub] # m * rr -> m * n
                        Vi= V[lb : ub, :] # rr * n -> m * n
                        Ui = np.hstack([Ui] * ((n-1)// delta + 1))[:,:n]
                        Vi = np.vstack([Vi] * ((m-1)// delta + 1))[:m,:]
                        if j==0:
                            W = Ui + Vi
                        else:
                            W *= Ui + Vi
                        cnt+=1
                    else:
                        break
                return np.linalg.matrix_rank(W)
            return lora_h_add

    try:   
        func = lora(type, r, rr)
        print(f"trainbale: {r * (m+n) / (m*n)}")
        for i in range(num_experiments):
            ranks[i] = func()
        print(f"exp rank: {ranks[0]}")
        print('----------------------')
        if verbose:
            # Display the histogram of ranks
            plt.hist(ranks,bins=range(max(1,-50+int(ranks[0]) ), 1+min(m,n,50+int(ranks[0]))), density=True)
            plt.xlabel('Rank')
            plt.ylabel('Frequency')
            plt.show()
    except Exception as e:
        print(e.__str__())
    
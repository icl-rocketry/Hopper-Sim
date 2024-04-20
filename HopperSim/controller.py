import numpy as np
import sys


def controllabilityCheck(A, B):    
    """
    This function checks the controllability of the system by checking the rank of the controllability matrix.
    ------------------------------------------------------------------------------------------------
    A: Dynamics matrix of the system (anonymously defined)
    B: Control matrix of the system (anonymously defined)
    ------------------------------------------------------------------------------------------------
    """
    C = B
    for i in range(1,len(A)):
        C = np.hstack((C, A**i @ B))

    rank = np.linalg.matrix_rank(C)
    size = len(A)
    
    if rank != size:
        print("Rank of Controllability Matrix: " + str(rank) + "\n"
              "Size of Matrix A: " + str(size) + "\n"
              "System is not controllable!")
        sys.exit()
        


def observerabilityCheck(A, C):
    """
    This function checks the observability of the system by checking the rank of the observability matrix.
    ------------------------------------------------------------------------------------------------
    A: Dynamics matrix of the system (anonymously defined)
    B: Control matrix of the system (anonymously defined)
    ------------------------------------------------------------------------------------------------
    """   
    O = C
    for i in range(1,len(A)):
        O = np.vstack((O, C @ (A**i)))

    rank = np.linalg.matrix_rank(O) # Not working rn
    size = len(A)
    
    if rank != size:
        print("Rank of Observability Matrix: " + str(rank) + "\n"
              "Size of Matrix A: " + str(size) + "\n"
              "System is not observable!")
        sys.exit()
        


def stabilityCheck(A):
    """
    NOTE: This is a stability check for LINEAR systems only -> Need to implement Lyapunov Stability for non-linear systems
    This function checks the stability of the system by checking the eigenvalues of the dynamics matrix.
    ------------------------------------------------------------------------------------------------
    A: Dynamics matrix of the system (anonymously defined)
    ------------------------------------------------------------------------------------------------
    """
    eig = np.linalg.eigvals(A)
    print(eig)
    for i in eig:
        if i.real >= 0:
            print("System is unstable!")
            sys.exit()
            
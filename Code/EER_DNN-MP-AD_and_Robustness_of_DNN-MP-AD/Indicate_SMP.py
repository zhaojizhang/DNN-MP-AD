import numpy as np

def Indic (M,Ns,Np):
    #weights initial
    mask_Y =np.zeros((M*Np,M*Ns*Np)).astype("float32")
    mask_D2A=np.zeros((M*Ns*Np,M*Ns*Np)).astype("float32")

    mask_sigma2V = np.ones((1, M * Ns * Np)).astype("float32")
    mask_H22A= np.ones((1, M * Ns * Np)).astype("float32")
    mask_lsp2B = np.ones((1, Ns * Np)).astype("float32")
    mask_logPa2C = np.ones((1, Ns * Np)).astype("float32")
    mask_lsp2D = np.ones((1,M * Ns * Np)).astype("float32")
    mask_lsp2dec = np.ones((1, Ns * Np)).astype("float32")

    mask_C2dec = np.ones((1, Ns * Np)).astype("float32")
    for i in range(0,Ns):     #mask_Y
        mask_Y[:,i*(M*Np):(i+1)*(M*Np)]=np.eye(M*Np)
    mask_A2B=np.kron(np.eye(Ns*Np),np.ones((M,1)).astype("float32"))

    mask_B2C=np.kron(np.eye(Ns),np.ones((Np,Np)).astype("float32"))-np.eye(Ns*Np)

    mask_C2D=np.kron(np.eye(Ns*Np),np.ones((1,M)).astype("float32"))

    mask_A2D=np.kron(np.eye(Ns*Np),np.ones((M,M)).astype("float32"))-np.eye(M*Ns*Np)


    for j in range(0,Ns):
        mask_D2A[j*(M*Np),0]=1
    for i in range(1,M*Ns*Np):
        mask_D2A[:,i]=np.roll(mask_D2A[:,0],i)
    mask_D2A=mask_D2A-np.eye(M*Ns*Np)
    mask_A2dec=np.kron(np.eye(Ns*Np),np.ones((M,1)).astype("float32"))

        
    return mask_Y,mask_A2B,mask_B2C,mask_C2D,mask_A2D,mask_D2A,mask_A2dec,mask_sigma2V,mask_H22A,mask_lsp2B, mask_logPa2C, mask_lsp2D, mask_lsp2dec, mask_C2dec


# mask_Y,mask_A2B,mask_B2C,mask_C2D,mask_A2D,mask_D2A,mask_A2dec,mask_C2dec=Indic(M=4,Ns=3,Np=2)
# print(mask_C2dec)
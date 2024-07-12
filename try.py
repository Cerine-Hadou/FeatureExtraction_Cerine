import numpy as np

def load_and_verify_signatures():
    try:
       
        glcm_signatures = np.load('glcm_signatures.npy', allow_pickle=True)
        print(f"GLCM signatures loaded: shape = {glcm_signatures.shape}")
        print("Example GLCM signature:", glcm_signatures[0])
        print("GLCM signature length:", len(glcm_signatures[0]))

       
        bit_signatures = np.load('bit_signatures.npy', allow_pickle=True)
        print(f"BIT signatures loaded: shape = {bit_signatures.shape}")
        print("Example BIT signature:", bit_signatures[0])
        print("BIT signature length:", len(bit_signatures[0]))

    except Exception as e:
        print(f"Error loading signatures: {e}")

if __name__ == "__main__":
    load_and_verify_signatures()

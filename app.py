from descriptor import glcm, bitdesc

path = 'image/test.png'

def main():
    
    feat_glcm = glcm(path)
    feat_bit = bitdesc(path)
    
    print(f'GLCM\n-----\n{feat_glcm}')
    print(f'BiT\n---\n{feat_bit}')

if __name__ == '__main__':
    main()
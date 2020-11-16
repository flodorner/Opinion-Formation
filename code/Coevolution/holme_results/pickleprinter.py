import pickle, sys
#print pickle content: python3 pickleprinter.py somethingsomething.pickle
def pickleprinter(filename):
    with open(filename, "rb") as f:
                x=pickle.load(f)
    print(x)

if __name__ == "__main__":  
    pickleprinter(sys.argv[1])
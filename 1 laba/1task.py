def main():
    deposite = float(input("Введите депозит: "))
    
    for i in range(5):
        deposite *= 1.08
        if i == 0 or i == 1 or i == 4:
            print('%.2f на конец %d года' % (deposite, i+1))

if __name__ == "__main__":
    main()
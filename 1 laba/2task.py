def main():
    mark = int(input("Введите оценку: "))
    if 0 <= mark <= 100:

        if mark <= 51:
            print("Неудовлетворительно")
        elif 52 <= mark <= 69:
            print("Удовлетворительно")
        elif 70 <= mark <= 84:
            print("Хорошо")
        elif 85 <= mark <= 100:
            print("Отлично")
    else:
        print("Введено некорректное число")


if __name__ == "__main__":
    main()
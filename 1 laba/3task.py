def main():
    sum = 0

    while True:
        price = input("Введите сумму: ")
        if price == "" or price == " ":
            break
        sum += int(price)

    print("Сумма всех введенных пользователем сумм: %.1f" % (sum))

    ans = 0

    if sum % 5 < 2.5:
        ans = sum - (sum % 5)
    else:
        ans = sum + (sum % 5)

    print("Надо заплатить: %.1f" % (ans))


if __name__ == "__main__":
    main()
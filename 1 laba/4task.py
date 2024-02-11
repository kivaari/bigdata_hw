def price(distance):
    base_rate = 380
    rate_per_100m = 50
    rd = (distance * 1000) / 100
    total_price = base_rate + rd * rate_per_100m
    print("Итоговая цена: %.2f" % (total_price))


def main():
    distance = float(input("Введите расстояние в километрах: "))
    price(distance)


if __name__ == "__main__":
    main()

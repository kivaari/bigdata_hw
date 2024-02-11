def find_keys_by_value(dictionary, target_value):
    keys_list = [key for key, value in dictionary.items() if value == target_value]
    return keys_list


def main():
    sample_dict = {'яблоко': 5, 'апельсин': 3, 'банан': 5, 'лимон': 2, 'виноград': 5}

    search_value = 5
    result_keys = find_keys_by_value(sample_dict, search_value)

    print(f"Словарь: {sample_dict}")
    print(f"Искомое значение: {search_value}")
    print(f"Соответствующие ключи: {result_keys}")


if __name__ == "__main__":
    main()

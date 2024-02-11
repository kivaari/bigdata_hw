def format_list(elements):
    if len(elements) == 0:
        return ""
    elif len(elements) == 1:
        return elements[0]
    else:
        formatted_elements = ", ".join(elements[:-1]) + " и " + elements[-1]
        return formatted_elements


def main():
    user_input = input("Введите элементы списка, разделяя их запятыми: ")

    elements_list = [element.strip() for element in user_input.split(',')]
    formatted_string = format_list(elements_list)

    print("Отформатированный список:", formatted_string)


if __name__ == "__main__":
    main()

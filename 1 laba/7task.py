def find_longest_word(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            words = [word.strip(".,?!\"':;()[]{}") for word in content.split()]

            if not words:
                print("Файл пуст.")
                return

            max_length = max(len(word) for word in words)
            longest_words = [word for word in words if len(word) == max_length]

            print(f"Длина самого длинного слова: {max_length}")
            print(f"Самые длинные слова: {', '.join(longest_words)}")

    except FileNotFoundError:
        print("Файл не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


def main():
    file_path = 'text.txt'
    find_longest_word(file_path)

if __name__ == "__main__":
    main()
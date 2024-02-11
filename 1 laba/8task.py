def find_chain(word, word_list):
    next_words = [next_word for next_word in word_list if next_word[0] == word[-1]]

    if not next_words:
        return [word]

    chains = [find_chain(next_word, [w for w in word_list if w != next_word]) for next_word in next_words]

    max_chain = max(chains, key=len, default=[])

    return [word] + max_chain

def main():
    word_list = []
    n = int(input("Введите количество слов: "))
    
    for i in range(n):
        word = input(f"Введите слово {i+1}: ")
        word_list.append(word.lower())

    start_word = input("Введите начальное слово: ").lower()

    result_chain = find_chain(start_word, word_list)

    print("Максимальная последовательность слов:")
    print(" -> ".join(result_chain))

if __name__ == "__main__":
    main()

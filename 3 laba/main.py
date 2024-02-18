import re


def phones(text):
    phone_pattern = r'\+\d\s\(\d{4}\)\s\d{2}-\d{2}-\d{2}'
    phones_found = re.findall(phone_pattern, text)
    return phones_found


def emails(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails_found = re.findall(email_pattern, text)
    return emails_found


def names(text):
    names_pattern = r'[А-ЯЁ][а-яё]+\s[А-ЯЁ]\.[А-ЯЁ]\.'
    names_found = re.findall(names_pattern, text)
    return names_found


def position(text):
    position_pattern = r'[А-Я][а-я -]+(?=[А-ЯЁ][а-яё]+\s[А-ЯЁ]\.[А-ЯЁ]\.)'
    positions_found = re.findall(position_pattern, text)
    return positions_found


def main():
    with open("text.txt", "r", encoding="utf-8") as file:
        string = file.read()
   
    print("Должность")
    print(position(string))
    print("Имя")
    print(names(string))
    print("Телефон")
    print(phones(string))
    print("Email")
    print(emails(string))


if __name__ == "__main__":
    main()



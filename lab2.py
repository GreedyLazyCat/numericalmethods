def start():
    print("Выберете метод:")
    print("1. Метод деление отрезка пополам")
    print("2. Метод Ньютона")
    choice = input().trim()
    match choice:
        case "1":
            print("1. Метод деление отрезка пополам")
        case "2":
            print("2. Метод Ньютона")
        case _:
            print("Такого варианта нет")
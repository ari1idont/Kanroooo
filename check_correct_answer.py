def right_answer(c,a):
    key=True
    #calc = input("Type calculation: ")
    expected_answer=str(eval(c))
    #print(expected_answer)
    #print(a)
    right=[str(d) for d in expected_answer]
    #print(right)
    print(right)
    for i in a:
        if str(i) in right:
            continue
        else:
            key=False
    #print(key)
    return key

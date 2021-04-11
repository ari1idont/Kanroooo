def right_answer_alpha(c,a):
    key=True
    right=[ str(d).upper() for d in c ]
    print(right)
    for i in a:
        if str(i) in right:
            continue
        else:
            key=False
    return key
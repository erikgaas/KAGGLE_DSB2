p1 = list(reversed([-70, -246]))
p2 = list(reversed([164, 367]))
p3 = list(reversed([238, 18]))
p4 = list(reversed([-101, 245]))



def calc_stupid_intersection(p1, p2, p3, p4):
    s1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
    print(s1)
    s2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
    print(s2)

    b1 = p2[1] - (s1 * p2[0])
    print(b1)
    b2 = p4[1] - (s2 * p4[0])
    print(b2)

    x_coord = (b2-b1) / (s1-s2)
    y_coord = x_coord*s1 + b1
    return (int(round(x_coord)), int(round(y_coord)))


print(p1, p2, p3, p4)

print(calc_stupid_intersection(p1, p2, p3, p4))
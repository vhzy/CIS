a=['1.jpg','2.jpg']
c = []
for b in a:
    c.append(int(b.split('.jpg')[0]))
print(c)
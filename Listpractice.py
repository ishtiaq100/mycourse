#list are mutable we can change the value of list
mobile = ['samsung',37.5,45000,True]
mobile.append("china")
mobile.insert(2,30.9)
mobile.remove(45000)
mobile.pop(0)
for mo in mobile:
    print(mo)
print(mobile)
print(type(mobile))
print(type(mobile[1]))
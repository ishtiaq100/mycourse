mobile = {'mobile':'samsung',"screen":37.5,"price":45000,"pta":True}
mobile["loc"]="pakistan"
mobile["mobile"]="vivo"
mobile.pop("pta")
#mobile.popitem()

#mobile.add("china")
#mobile.discard(45000)
for mo in mobile:
    print(mo,mobile[mo])
    #print(mobile[mo])
print(mobile)
print(type(mobile))
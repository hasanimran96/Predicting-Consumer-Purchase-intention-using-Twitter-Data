li = ["class","name","email"]
if(set(["class","text"]).issubset(set(li))): 
    print("proper subset")
else:
    print("not subset")    

str = ['Fndmemo', 'Python','TUtorial']
temp = [x.lower() for x in str]    
print(temp)
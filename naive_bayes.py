import pandas as pd

df = pd.read_csv("AnnotatedData.csv")

#print(df.at[5,'Purchase Intention'])

for count, row in df.iterrows():
    if(count, row['Purchase Intention'])=="undefined":
        print(df.drop([count, row['Purchase Intention']]),inplace=True)

print(df)

columns = ['id','text', 'class label']
#df2 = pd.DataFrame(index=id, columns=columns)

#print(df)

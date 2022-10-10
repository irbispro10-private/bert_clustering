import json

from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
import csv
titles=[]
for_test=[]
v = DictVectorizer(sparse=False)
with open('test.csv', encoding='utf-8') as f:
    csv_reader = csv.reader(f, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            titles.append(json.loads(row[4]))
            for_test.append(row[2])
            line_count += 1

X = v.fit_transform(titles)
# print(X[0])
num_clusters=7
km = KMeans(n_clusters=num_clusters, n_init=1000)
km.fit(X)
clusters = km.labels_.tolist()

for i in range(num_clusters):
    print("cluster "+ str(i))
    for z, j in enumerate(clusters):
        if i==j:
            print(for_test[z])
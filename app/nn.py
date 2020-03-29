import pandas as pd
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib


hospital = pd.read_csv("hospital.csv",encoding='cp932')

is_complete_duplicate_keep_first = (hospital.duplicated(keep='first',subset=['医療機関名']))

new_hospital=hospital[~is_complete_duplicate_keep_first]
new_hospital.count()

hospital_name=new_hospital.loc[:,"医療機関名"]
#print(hospital_name)

hospital_URL=new_hospital.loc[:,"病院URL"]
#print(hospital_URL)

hospital_data = new_hospital.drop(["調査日",'医療機関名',"病院URL"],axis=1)
hospital_data.head()

data1 = pd.DataFrame(hospital_data)
new_data1 = pd.get_dummies(data1)
new_data1.head()


knn = NearestNeighbors(n_neighbors=9, algorithm="brute", metric="cosine")
model_knn=knn.fit(new_data1)


joblib.dump(knn, "nn.pkl", compress=True)

hospital_status=0
distance, indice = model_knn.kneighbors(new_data1.iloc[new_data1.index== hospital_status].values.reshape(1,-1),n_neighbors=11)
#distance, indice = model_knn.kneighbors(new_data1.iloc[new_data1.index== hospital_status].values.reshape(1,-1),n_neighbors=11)

#上位10個表示
for i in range(0, len(distance.flatten())):
    if  i == 0:
        pass
	#print('Recommendations if you like the hospital {0}:\n'.format(new_data1[new_data1.index== hospital_status].index[0]))
    else:
        index=new_data1.index[indice.flatten()[i]]    
        print('{0}　 \n病院名：{1}　\n病院URL：{2}\n'.format(i,hospital_name[index],hospital_URL[index],distance.flatten()[i]))
        #print('{0}: {1} with distance: {2}'.format(i,new_data1.index[indice.flatten()[i]],distance.flatten()[i]))
 
 
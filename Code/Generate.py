import numpy as np
import random
import pandas as pd

# Learner
from sklearn.neighbors import NearestNeighbors as NN

#Function to find the K nearest neighhours
def get_ngbr(df, knn):
    #np.random.seed(0)
    rand_sample_idx = random.randint(0, df.shape[0] - 1)
    parent_candidate = df.iloc[rand_sample_idx]
    distance,ngbr = knn.kneighbors(parent_candidate.values.reshape(1,-1),3,return_distance=True)    
    candidate_1 = df.iloc[ngbr[0][1]]    
    candidate_2 = df.iloc[ngbr[0][2]]    
    return distance,parent_candidate,candidate_1,candidate_2


def fair_generate_samples(no_of_samples,df,df_name,X_train,y_train,protected_attribute):

    #--------------------------------------------------------------------------------------------------
    #Calling function to find the KNN
    total_data = df.values.tolist()
    knn = NN(n_neighbors=5,algorithm='auto').fit(df)

    column_name=df.columns.tolist()
    #added by own
    #new_candidate_df=pd.DataFrame(columns=column_name)
    #added by own  end

    #--------------------------------------------------------------------------------------------------------------
    #Logic to create synthetic data

    for _ in range(no_of_samples):
    
        f = .3
        distance,parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)      
        mutant = []
        for key,value in parent_candidate.items():        
            #x1=distance[0][0]  
            x1=distance[0][1] 
            x2=distance[0][2] 
            x3=abs(x2-x1)
                
            if isinstance(parent_candidate[key], (bool, str)):
                if x1 <=x3:
                    mutant.append(np.random.choice([parent_candidate[key], child_candidate_1[key]]))   
                else:
                    mutant.append(np.random.choice([child_candidate_1[key], child_candidate_2[key]]))                      
            else:             
                if x1 <= x3:
                    mutant.append(parent_candidate[key] + f * (child_candidate_1[key] - parent_candidate[key]))
                else:
                    mutant.append(abs(child_candidate_1[key] + f * (child_candidate_2[key] - child_candidate_1[key])))
        total_data.append(mutant)
   
    final_df = pd.DataFrame(total_data)
    #--------------------------------------------------------------------------------------------------------------
    #Rename dataframe columns
    final_df.set_axis(column_name, axis=1,inplace=True)

    return final_df
 

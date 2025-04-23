From Generate import * 

    # %%
def situation(clf,X_train,y_train,keyword):
    #they have used  as a classifier
    X_flip = X_train.copy()
    X_flip[keyword] = np.where(X_flip[keyword]==1, 0, 1)
    a = np.array(clf.predict(X_train))
    b = np.array(clf.predict(X_flip))
    same = (a==b)
    #print(same) #[True  True  True ... False  True  True  True]
    same = [1 if each else 0 for each in same]  #[1 1 1... 0 1 1 1] if true makes it 1 ,else 0
    X_train['same'] = same #make a new column 'same' and put above list into it.
    X_train['y'] = y_train #make a new column 'y' and put y_train value into it.
    X_rest = X_train[X_train['same']==1] #This creates a new DataFrame (X_rest) that contains only the rows where the 'same' column is 1.
    y_rest = X_rest['y']
    X_rest = X_rest.drop(columns=['same','y'])

    print("Removed Points:",np.round((X_train.shape[0] - X_rest.shape[0]) / X_train.shape[0] * 100,4),"% || ", X_train.shape[0]-X_rest.shape[0])
    point_removed=np.round((X_train.shape[0] - X_rest.shape[0]) / X_train.shape[0] * 100,4)
    
    return X_rest,y_rest,point_removed


def fair_generate(random_state,dataset_orig_train1,dataset_orig_test1,X_train1,y_train1,X_test1,y_test1,protected_attribute,global_timestamp,folder_name,dataset_name):

    start_time = time.time()

    dataset_orig_train=copy.deepcopy(dataset_orig_train1)
    dataset_orig_test=copy.deepcopy(dataset_orig_test1)
    X_train=copy.deepcopy(X_train1)
    y_train=copy.deepcopy(y_train1)
    X_test=copy.deepcopy(X_test1)
    y_test=copy.deepcopy(y_test1)

    fair_generate_results={} #for storing all results.
  
    fair_generate_lgr={}
    fair_generate_svm={}
    fair_generate={}
    fair_generate_nb={}
    fair_generate_mlp={}
    fair_generate_dt={}
    fair_generate_knn={}

    print("Situation Testing......")
    clf1 = LogisticRegression(random_state=random_seed)
    clf1.fit(X_train, y_train)
    X_train, y_train,before_point_removed = situation(clf1, X_train, y_train, protected_attribute) #dataset is changing. 
    
    clf2 = LogisticRegression(random_state=random_seed)
    clf2.fit(X_train, y_train)

    dataset_orig_train=X_train
    dataset_orig_train['Probability']=y_train

    # first one is class value and second one is protected attribute value
    zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
    zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
    one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
    one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

    print(zero_zero,zero_one,one_zero,one_one)

    maximum = max(zero_zero,zero_one,one_zero,one_one)
         
    zero_zero_to_be_increased = maximum - zero_zero ## where class is 0 attribute is 0
    zero_one_to_be_increased = maximum - zero_one ## where class is 0 attribute is 1
    one_zero_to_be_increased = maximum - one_zero ## where class is 1 attribute is 0
    one_one_to_be_increased = maximum - one_one ## where class is 1 attribute is 1
    
    df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
    df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]
    df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
    df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]

    if zero_zero_to_be_increased==0:
        df_zero_one[protected_attribute] = df_zero_one[protected_attribute].astype(str)
        df_one_zero[protected_attribute] = df_one_zero[protected_attribute].astype(str)
        df_one_one[protected_attribute] = df_one_one[protected_attribute].astype(str)

        #calling fair_generate_samples
        df_zero_one = fair_generate_samples(zero_one_to_be_increased,df_zero_one,dataset_name,X_train,y_train,protected_attribute)
        df_one_zero = fair_generate_samples(one_zero_to_be_increased,df_one_zero,dataset_name,X_train,y_train,protected_attribute)
        df_one_one = fair_generate_samples(one_one_to_be_increased,df_one_one,dataset_name,X_train,y_train,protected_attribute)
  
        #appending dataframes
        df = df_one_zero.append(df_zero_one)
        df = df.append(df_one_one)
        df[protected_attribute] = df[protected_attribute].astype(float)

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df = df.append(df_zero_zero)

    
    elif zero_one_to_be_increased==0:
        df_zero_zero[protected_attribute] = df_zero_zero[protected_attribute].astype(str)
        df_one_zero[protected_attribute] = df_one_zero[protected_attribute].astype(str)
        df_one_one[protected_attribute] = df_one_one[protected_attribute].astype(str)

        #calling fair_generate_samples
        df_zero_zero = fair_generate_samples(zero_zero_to_be_increased,df_zero_zero,dataset_name,X_train,y_train,protected_attribute)
        df_one_zero = fair_generate_samples(one_zero_to_be_increased,df_one_zero,dataset_name,X_train,y_train,protected_attribute)
        df_one_one = fair_generate_samples(one_one_to_be_increased,df_one_one,dataset_name,X_train,y_train,protected_attribute)

        #appending dataframes
        df = df_one_zero.append(df_zero_zero)
        df = df.append(df_one_one)
        df[protected_attribute] = df[protected_attribute].astype(float)
        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df = df.append(df_zero_one)
    
    
    elif one_zero_to_be_increased==0:
        
        df_zero_one[protected_attribute] = df_zero_one[protected_attribute].astype(str)
        df_zero_one[protected_attribute] = df_zero_one[protected_attribute].astype(str)
        df_one_one[protected_attribute] = df_one_one[protected_attribute].astype(str)

        #calling fair_generate_samples
        df_zero_zero = fair_generate_samples(zero_zero_to_be_increased,df_zero_zero,dataset_name,X_train,y_train,protected_attribute)
        df_zero_one = fair_generate_samples(zero_one_to_be_increased,df_zero_one,dataset_name,X_train,y_train,protected_attribute)
        df_one_one = fair_generate_samples(one_one_to_be_increased,df_one_one,dataset_name,X_train,y_train,protected_attribute)

        #appending dataframes
        df = df_zero_one.append(df_zero_zero)
        df = df.append(df_one_one)
        df[protected_attribute] = df[protected_attribute].astype(float)
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df = df.append(df_one_zero)
    
    
    elif one_one_to_be_increased==0:
        
        df_zero_zero[protected_attribute] = df_zero_zero[protected_attribute].astype(str)
        df_zero_one[protected_attribute] = df_zero_one[protected_attribute].astype(str)
        df_one_zero[protected_attribute] = df_one_zero[protected_attribute].astype(str)
         
        #calling fair_generate_samples
        df_zero_zero = fair_generate_samples(zero_zero_to_be_increased,df_zero_zero,dataset_name,X_train,y_train,protected_attribute)
        df_zero_one = fair_generate_samples(zero_one_to_be_increased,df_zero_one,dataset_name,X_train,y_train,protected_attribute)
        df_one_zero = fair_generate_samples(one_zero_to_be_increased,df_one_zero,dataset_name,X_train,y_train,protected_attribute)
        
        #appending dataframes
        df = df_zero_one.append(df_zero_zero)
        df = df.append(df_one_zero)
        df[protected_attribute] = df[protected_attribute].astype(float)
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]
        df = df.append(df_one_one)
  
    df = df.reset_index(drop=True)
    
    X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
    
    #fair-situation testing
    X_train, y_train,after_point_removed = situation(clf2, X_train, y_train, protected_attribute) #dataset is changing. 

    df=copy.deepcopy(X_train)
    df['Probability']=y_train

    filename='generated_data/'+folder_name+'/'+str(global_timestamp)+'/fairgenerate'+str(random_seed)+'.csv'
    df.to_csv(filename,index=False)
   
    # Prepare dataset_t for metrics calculation
    dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                   unfavorable_label=0.0,
                                   df=dataset_orig_test,
                                   label_names=['Probability'],
                                   protected_attribute_names=[protected_attribute])

    added_to_all = time.time()- start_time 

    
    # Logistic Regression
    start_time = time.time()
    clf_lr = LogisticRegression(random_state=random_seed)
    clf_lr.fit(X_train, y_train)
    fair_generate_results['LogisticRegression'] = calculate_metrics(clf_lr, X_test, y_test, dataset_t, protected_attribute)
    fair_generate_results['LogisticRegression']['before_situation_testing']=before_point_removed
    fair_generate_results['LogisticRegression']['fair_situation_testing']=after_point_removed
    fair_generate_results['LogisticRegression']['time'] = np.round(time.time() - start_time + added_to_all,4)
    fair_generate_results['LogisticRegression']['added_time']=np.round(added_to_all,4)

    # Random Forest
    start_time = time.time()
    clf_rf = RandomForestClassifier(random_state=random_seed)
    clf_rf.fit(X_train, y_train)
    fair_generate_results['RandomForest'] = calculate_metrics(clf_rf, X_test, y_test, dataset_t, protected_attribute)
    fair_generate_results['RandomForest']['time'] = np.round(time.time() - start_time + added_to_all,4)
    
    # Decision Tree
    start_time = time.time()
    clf_dt = DecisionTreeClassifier(random_state=random_seed)
    clf_dt.fit(X_train, y_train)
    fair_generate_results['DecisionTree'] = calculate_metrics(clf_dt, X_test, y_test, dataset_t, protected_attribute)
    fair_generate_results['DecisionTree']['time'] = np.round(time.time() - start_time + added_to_all,4)

     # Initialize a sequential model
    start_time = time.time()
    clf_deep_learning = Sequential()
    clf_deep_learning.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # Correct input shape for 5 features
    clf_deep_learning.add(Dense(32, activation='relu'))     # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(16, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(8, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(4, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(1, activation='sigmoid'))  # Replace output_units with the number of output classes
    clf_deep_learning.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])     # Compile the model
    clf_deep_learning.fit(X_train, y_train,epochs=20)                 # Train (fit) the model on the training data
    fair_generate_results['DeepLearning1'] = calculate_metrics_deeplearning(clf_deep_learning, X_test, y_test, dataset_t, protected_attribute)
    fair_generate_results['DeepLearning1']['time'] = np.round(time.time() - start_time + added_to_all,4)


       # Initialize a sequential model
    start_time = time.time()
    clf_deep_learning = Sequential()
    clf_deep_learning.add(Dense(50, activation='relu', input_shape=(X_train.shape[1],)))  # Correct input shape for 5 features
    clf_deep_learning.add(Dense(30, activation='relu'))     # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(15, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(10, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(5, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(1, activation='sigmoid'))  # Replace output_units with the number of output classes
    clf_deep_learning.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])     # Compile the model
    clf_deep_learning.fit(X_train, y_train,epochs=20)                 # Train (fit) the model on the training data
    fair_generate_results['DeepLearning2'] = calculate_metrics_deeplearning(clf_deep_learning, X_test, y_test, dataset_t, protected_attribute)
    fair_generate_results['DeepLearning2']['time'] = np.round(time.time() - start_time + added_to_all,4)


     # Initialize a sequential model
    start_time = time.time()
    clf_deep_learning = Sequential()
    clf_deep_learning.add(Dense(30, activation='relu', input_shape=(X_train.shape[1],)))  # Correct input shape for 5 features
    clf_deep_learning.add(Dense(20, activation='relu'))     # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(15, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(10, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(5, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(1, activation='sigmoid'))  # Replace output_units with the number of output classes
    clf_deep_learning.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])     # Compile the model
    clf_deep_learning.fit(X_train, y_train,epochs=20)                 # Train (fit) the model on the training data
    fair_generate_results['DeepLearning3'] = calculate_metrics_deeplearning(clf_deep_learning, X_test, y_test, dataset_t, protected_attribute)
    fair_generate_results['DeepLearning3']['time'] = np.round(time.time() - start_time + added_to_all,4)


     # Initialize a sequential model
    start_time = time.time()
    clf_deep_learning = Sequential()
    clf_deep_learning.add(Dense(30, activation='relu', input_shape=(X_train.shape[1],)))  # Correct input shape for 5 features
    clf_deep_learning.add(Dense(20, activation='relu'))     # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(15, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(15, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(10, activation='relu'))    # Add hidden layers with specified number of units
    clf_deep_learning.add(Dense(1, activation='sigmoid'))  # Replace output_units with the number of output classes
    clf_deep_learning.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])     # Compile the model
    clf_deep_learning.fit(X_train, y_train,epochs=20)                 # Train (fit) the model on the training data
    fair_generate_results['DeepLearning4'] = calculate_metrics_deeplearning(clf_deep_learning, X_test, y_test, dataset_t, protected_attribute)
    fair_generate_results['DeepLearning4']['time'] = np.round(time.time() - start_time + added_to_all,4)

    
    return fair_generate_results
